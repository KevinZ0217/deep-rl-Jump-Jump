import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
from GetEnv import GetEnv
import tensorflow as tf
import tensorflow_probability as tfp
import copy
from PPOMemory import PPOMemory
from scipy import signal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MemoryPPO:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    This is adapted version of the Spinning Up Open Ai PPO code of the buffer.
    https://github.com/openai/spinningup/blob/master/spinup/algos/ppo/ppo.py
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        # a fucntion so that different dimensions state array shapes are all processed corecctly
        def combined_shape(length, shape=None):
            if shape is None:
                return (length,)
            return (length, shape) if np.isscalar(shape) else (length, *shape)
        # just empty arrays with appropriate sizes
        self.obs_buf = np.zeros(combined_shape(
            size, obs_dim), dtype=np.float32)  # states
        self.act_buf = np.zeros(combined_shape(
            size, act_dim), dtype=np.float32)  # actions
        # actual rwards from state using action
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # predicted values of state
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)  # gae advantewages
        self.ret_buf = np.zeros(size, dtype=np.float32)  # discounted rewards
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        example input: [x0, x1, x2] output: [x0 + discount * x1 + discount^2 * x2,  x1 + discount * x2, x2]
        """
        return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def store(self, obs, act, rew, val):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.ptr += 1

    def finish_path(self, last_val=0):
        """Finishes an episode of data collection by calculating the diffrent rewards and resetting pointers.
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(
            deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get_batch(self, batch_size):
        """simply retuns a randomized batch of batch_size from the data in memory
        """
        # make a randlim list with batch_size numbers.
        pos_lst = np.random.randint(self.ptr, size=batch_size)
        return self.obs_buf[pos_lst], self.act_buf[pos_lst], self.adv_buf[pos_lst], self.ret_buf[pos_lst], self.val_buf[pos_lst]

    def clear(self):
        """Set back pointers to the beginning
        """
        self.ptr, self.path_start_idx = 0, 0


class ActorNetwork(tf.keras.Model):
    """
    输出动作：在这里就是按压时长秒数
    """

    def __init__(self):
        super().__init__()

        self.average_polling_2d = tf.keras.layers.AveragePooling2D((3, 3), strides=(2, 2), padding='same')
        self.conv2d_1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(2, 2), padding='same')
        self.activation_1 = tf.keras.layers.Activation('relu')

        self.conv2d_2 = tf.keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same')
        self.activation_2 = tf.keras.layers.Activation('relu')

        self.conv2d_3 = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')
        self.activation_3 = tf.keras.layers.Activation('relu')

        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(200, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(200, activation='relu')

        # self.outputs = tf.keras.layers.Dense(1, activation='tanh')
        self.mu = tf.keras.layers.Dense(1, activation='tanh')
        self.sigma = tf.keras.layers.Dense(1, activation='softplus')

    def call(self, inputs, training=False):
        print("inputs: ", inputs.shape)
        x = inputs[:, :, :, :]

        x = self.average_polling_2d(x)

        x = self.conv2d_1(x)
        x = self.activation_1(x)

        x = self.conv2d_2(x)
        x = self.activation_2(x)

        x = self.conv2d_3(x)
        x = self.activation_3(x)

        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        # outputs = self.outputs(x)
        outputs = (mu, sigma)
        return outputs


class CriticNetwork(tf.keras.Model):
    """
    输入：next_state, next_action
    输出：评估其状态的价值
    """

    def __init__(self):
        super().__init__()

        self.average_polling_2d = tf.keras.layers.AveragePooling2D((3, 3), strides=(2, 2), padding='same')
        self.conv2d_1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(2, 2), padding='same')
        self.activation_1 = tf.keras.layers.Activation('relu')

        self.conv2d_2 = tf.keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same')
        self.activation_2 = tf.keras.layers.Activation('relu')

        self.conv2d_3 = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')
        self.activation_3 = tf.keras.layers.Activation('relu')

        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(200, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(200, activation='relu')

        # input action
        self.dense_action_1 = tf.keras.layers.Dense(200, activation='relu')
        self.add_action_1 = tf.keras.layers.Add()

        # final
        self.outputs = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs_state, input_action, training=False):
        # input state
        x = inputs_state[:, :, :, :]

        x = self.average_polling_2d(x)

        x = self.conv2d_1(x)
        x = self.activation_1(x)

        x = self.conv2d_2(x)
        x = self.activation_2(x)

        x = self.conv2d_3(x)
        x = self.activation_3(x)

        x = self.flatten(x)
        x_s = self.dense_1(x)

        # input action
        x_a = self.dense_action_1(input_action)

        x = self.add_action_1([x_s, x_a])

        x = self.dense_2(x)
        outputs = self.outputs(x)

        return outputs


class Agent:
    """
    function:
    初始化网络
        1.网络预测
        2.回放经验计算loss
    """

    def __init__(self, env, TRAJECTORY_BUFFER_SIZE):
        self.env = env
        # 记录
        self.memory = []

        self.gamma = 0.99
        self.TAU = 0.001  # Target Network HyperParameters
        self.lr_actor = 0.0001  # Learning rate for Actor
        self.lr_critic = 0.001  # Lerning rate for Critic
        self.TRAJECTORY_BUFFER_SIZE = TRAJECTORY_BUFFER_SIZE
        self.episode_count = 2000
        self.max_steps = 100000
        self.reward = 0
        self.done = False
        self.step = 0
        self.epsilon = 1
        self.epsilon_decay = .995
        self.epsilon_min = 0.1
        self.indicator = 0
        self.TARGET_UPDATE_ALPHA = 0.95
        self.GAMMA = 0.99
        self.GAE_LAMBDA = 0.95
        self.CLIPPING_LOSS_RATIO = 0.1
        self.ENTROPY_LOSS_RATIO = 0.001
        self.TARGET_UPDATE_ALPHA = 0.9
        self.NOISE = 1.0  # Exploration noise, for continous action space
        self.batch_size = 32

        self.mem_len = 200 # TRAJECTORY_BUFFER_SIZE?

        self.clip_pram = 0.2

        self.actor_optimizer = tf.keras.optimizers.Adam(self.lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.lr_critic)
        self.memory = MemoryPPO(
            (1, 4), 1, self.TRAJECTORY_BUFFER_SIZE) # state_dim to be fixed
        # 创建4个网络
        self.actor = ActorNetwork()
        self.actor_old = ActorNetwork()
        self.actor_old.set_weights(self.actor.get_weights())
        self.actor_target = ActorNetwork()
        self.critic = CriticNetwork()
        self.critic_target = CriticNetwork()
        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediciton = np.zeros((1, 2 * 1))


    def act(self, state):
        mu, sigma = self.actor(state)
        mu = float(mu.numpy())
        sigma = float(sigma.numpy())
        print("mu:", mu)
        print("sigma:", sigma)
        action = np.random.normal(loc=mu, scale=sigma, size=1)
        print("action:", action)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        action = tf.reshape(action, (1, 1))
        return action

    def update_tartget_network(self):
        """Softupdate of the target network.
        In ppo, the updates of the
        """
        alpha = self.TARGET_UPDATE_ALPHA
        actor_weights = np.array(self.actor.get_weights())
        actor_tartget_weights = np.array(self.actor_old.get_weights())
        new_weights = alpha * actor_weights + (1 - alpha) * actor_tartget_weights
        self.actor_old.set_weights(new_weights)

    def ppo_loss(self, advantage, old_prediction):
        """The PPO custom loss.
        For explanation see for example:
        https://youtu.be/WxQfQW48A4A
        https://youtu.be/5P7I-xPq8u8
        Log Probability of  loss: (x-mu)²/2sigma² - log(sqrt(2*PI*sigma²))
        entropy of normal distribution: sqrt(2*PI*e*sigma²)
        params:
            :advantage: advantage, needed to process algorithm
            :old_predictioN: prediction from "old" network, needed to process algorithm
        returns:
            :loss: keras type loss fuction (not a value but a fuction with two parameters y_true, y_pred)
        TODO:
            probs = tf.distributions.Normal(mu,sigma)
            probs.sample #choses action
            probs.prob(action) #probability of action
            https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/12_Proximal_Policy_Optimization/simply_PPO.py
        """

        def get_log_probability_density(network_output_prediction, y_true):
            """Sub-function to get the logarithmic probability density.
            expects the prediction (containing mu and sigma) and the true action y_true
            Formula for pdf and log-pdf see https://en.wikipedia.org/wiki/Normal_distribution
            """
            # the actor output contains mu and sigma concatenated. split them. shape is (batches,2xaction_n)
            mu_and_sigma = network_output_prediction
            mu = mu_and_sigma[:, 0:self.action_n]
            sigma = mu_and_sigma[:, self.action_n:]
            variance = tf.keras.backend.square(sigma)
            pdf = 1. / tf.kerasbackend.sqrt(2. * np.pi * variance) * tf.keras.backend.exp(
                -tf.keras.backend.square(y_true - mu) / (2. * variance))
            log_pdf = tf.keras.backend.log(pdf + tf.keras.backend.epsilon())
            return log_pdf

        # refer to Keras custom loss function intro to understand why we define a funciton inside a function.
        # here y_true are the actions taken and y_pred are the predicted prob-distribution(mu,sigma) for each n in acion space
        def loss(y_true, y_pred):
            # First the probability density function.
            log_probability_density_new = get_log_probability_density(y_pred, y_true)
            log_probability_density_old = get_log_probability_density(old_prediction, y_true)
            # Calc ratio and the surrogates
            # ratio = prob / (old_prob + K.epsilon()) #ratio new to old
            ratio = tf.keras.backend.exp(log_probability_density_new - log_probability_density_old)
            surrogate1 = ratio * advantage
            clip_ratio = tf.keras.backend.clip(ratio, min_value=1 - self.CLIPPING_LOSS_RATIO,
                                        max_value=1 + self.CLIPPING_LOSS_RATIO)
            surrogate2 = clip_ratio * advantage
            # loss is the mean of the minimum of either of the surrogates
            loss_actor = - tf.keras.backend.mean(tf.keras.backend.minimum(surrogate1, surrogate2))
            # entropy bonus in accordance with move37 explanation https://youtu.be/kWHSH2HgbNQ
            sigma = y_pred[:, self.action_n:]
            variance = tf.keras.backend.square(sigma)
            loss_entropy = self.ENTROPY_LOSS_RATIO * tf.keras.backend.mean(
                -(tf.keras.backend.log(2 * np.pi * variance) + 1) / 2)  # see move37 chap 9.5
            # total bonus is all losses combined. Add MSE-value-loss here as well?
            return loss_actor + loss_entropy

        return loss


    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
    def update(self, state, action, reward):
        self.actor.set_weights(self.model)

    def test_reward(self, env):
        total_reward = 0
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(agentoo7.actor(np.array([state])).numpy())
            next_state, reward, done= env.touch_in_step(action)
            state = next_state
            total_reward += reward

        return total_reward


if __name__ == "__main__":
    env = GetEnv()
    tf.random.set_seed(336699)
    agentoo7 = Agent(env)
    steps = 50
    ep_reward = []
    total_avgr = []
    target = False
    best_reward = 0
    avg_rewards_list = []

    for e in range(1000):
        if target == True:
            break

        done = False
        state = env.reset()
        all_aloss = []
        all_closs = []
        rewards = []
        states = []
        actions = []
        probs = []
        dones = []
        values = []
        print("new episode")

        for s in range(steps):


            #print('state shape:', state.shape)
            action = agentoo7.act(state)
            #value = agentoo7.critic(np.array([state])).numpy()
            print("action:", action)
            next_state, reward, done = env.touch_in_step(action)
            dones.append(1 - done)
            rewards.append(reward)
            states.append(state)
            # actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
            actions.append(action)
            #prob = agentoo7.actor(np.array([state]))
            #probs.append(prob[0])
            #values.append(value[0][0])
            state = copy.deepcopy(next_state)
            print("next_state:", next_state)
            if done:
                break

        value = agentoo7.critic([state])
        # values.append(value[0][0])
        # np.reshape(probs, (len(probs), 2))
        # #probs = np.stack(probs, axis=0)
        #
        # states, actions, returns, adv = preprocess1(states, actions, rewards, dones, values, 1)
        #
        # for epocs in range(10):
        #     al, cl = agentoo7.learn(states, actions, adv, probs, returns)
        #     # print(f"al{al}")
        #     # print(f"cl{cl}")
        #
        # avg_reward = np.mean([test_reward(env) for _ in range(5)])
        # print(f"total test reward is {avg_reward}")
        # avg_rewards_list.append(avg_reward)
        # if avg_reward > best_reward:
        #     print('best reward=' + str(avg_reward))
        #     agentoo7.actor.save('model_actor_{}_{}'.format(s, avg_reward), save_format="tf")
        #     agentoo7.critic.save('model_critic_{}_{}'.format(s, avg_reward), save_format="tf")
        #     best_reward = avg_reward
        # if best_reward == 200:
        #     target = True
        # env.reset()
