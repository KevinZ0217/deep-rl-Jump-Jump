import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import GetEnv
import tensorflow as tf
import tensorflow_probability as tfp

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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

        self.outputs = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, inputs, training=False):
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

        outputs = self.outputs(x)

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

    def __init__(self, env):
        self.env = env
        # 记录
        self.memory = []

        self.gamma = 0.99
        self.TAU = 0.001  # Target Network HyperParameters
        self.lr_actor = 0.0001  # Learning rate for Actor
        self.lr_critic = 0.001  # Lerning rate for Critic

        self.episode_count = 2000
        self.max_steps = 100000
        self.reward = 0
        self.done = False
        self.step = 0
        self.epsilon = 1
        self.epsilon_decay = .995
        self.epsilon_min = 0.1
        self.indicator = 0

        self.batch_size = 32

        self.mem_len = 200

        self.clip_pram = 0.2

        self.actor_optimizer = tf.keras.optimizers.Adam(self.lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.lr_critic)

        # 创建4个网络
        self.actor = ActorNetwork()
        self.actor_target = ActorNetwork()
        self.critic = CriticNetwork()
        self.critic_target = CriticNetwork()

    def act(self, state):
        prob = self.actor(state)
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()

        return int(action.numpy()[0])

    def test_reward(self, env):
        # total_reward = 0
        # state = env.reset()
        # done = False
        # while not done:
        #     action = np.argmax(agentoo7.actor(np.array([state])).numpy())
        #     next_state, reward, done, _ = env.step(action)
        #     state = next_state
        #     total_reward += reward

        # return total_reward
        pass

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

    for s in range(steps):
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
        print("new episod")

        for e in range(128):

            action = agentoo7.act(state)
            value = agentoo7.critic(np.array([state])).numpy()
            next_state, reward, done, _ = env.step(action)
            dones.append(1 - done)
            rewards.append(reward)
            states.append(state)
            # actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
            actions.append(action)
            prob = agentoo7.actor(np.array([state]))
            probs.append(prob[0])
            values.append(value[0][0])
            state = next_state
            if done:
                env.reset()

        value = agentoo7.critic(np.array([state])).numpy()
        values.append(value[0][0])
        np.reshape(probs, (len(probs), 2))
        probs = np.stack(probs, axis=0)

        states, actions, returns, adv = preprocess1(states, actions, rewards, dones, values, 1)

        for epocs in range(10):
            al, cl = agentoo7.learn(states, actions, adv, probs, returns)
            # print(f"al{al}")
            # print(f"cl{cl}")

        avg_reward = np.mean([test_reward(env) for _ in range(5)])
        print(f"total test reward is {avg_reward}")
        avg_rewards_list.append(avg_reward)
        if avg_reward > best_reward:
            print('best reward=' + str(avg_reward))
            agentoo7.actor.save('model_actor_{}_{}'.format(s, avg_reward), save_format="tf")
            agentoo7.critic.save('model_critic_{}_{}'.format(s, avg_reward), save_format="tf")
            best_reward = avg_reward
        if best_reward == 200:
            target = True
        env.reset()
