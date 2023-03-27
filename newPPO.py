import tensorflow as tf
import numpy as np
import random
from GetEnv import GetEnv
import tensorflow_probability as tfp
import os
import pickle
file_memory = "./data/ppo_mem.p"
file_score = "./data/ppo_score.p"



class ActorNetwork(tf.keras.Model):
    def __init__(self, action_dim):
        super(ActorNetwork, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(200, 120, 1))

        self.average_pooling_2d = tf.keras.layers.AveragePooling2D((3, 3), strides=(2, 2), padding='same')
        self.conv2d_1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(2, 2), padding='same')
        self.activation_1 = tf.keras.layers.Activation('relu')

        self.conv2d_2 = tf.keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same')
        self.activation_2 = tf.keras.layers.Activation('relu')

        self.conv2d_3 = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')
        self.activation_3 = tf.keras.layers.Activation('relu')

        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(200, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(200, activation='relu')

        self.mean = tf.keras.layers.Dense(action_dim, activation='linear')
        self.log_std = tf.keras.layers.Dense(action_dim, activation='linear')
    def call(self, inputs, training=True):
        inputs = tf.reshape(inputs, (-1, 200, 120, 1))

        x = inputs[:, :, :, :]

        x = self.average_pooling_2d(x)

        x = self.conv2d_1(x)
        x = self.activation_1(x)

        x = self.conv2d_2(x)
        x = self.activation_2(x)

        x = self.conv2d_3(x)
        x = self.activation_3(x)

        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)

        mean = self.mean(x)
        log_std = self.log_std(x)
        std = tf.nn.softplus(log_std)

        return mean, std


class CriticNetwork(tf.keras.Model):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(200, 120, 1))

        self.average_pooling_2d = tf.keras.layers.AveragePooling2D((3, 3), strides=(2, 2), padding='same')
        self.conv2d_1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(2, 2), padding='same')
        self.activation_1 = tf.keras.layers.Activation('relu')

        self.conv2d_2 = tf.keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same')
        self.activation_2 = tf.keras.layers.Activation('relu')

        self.conv2d_3 = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')
        self.activation_3 = tf.keras.layers.Activation('relu')

        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(200, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(200, activation='relu')

        self.outputs = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs, training=True):
        x = inputs[:, :, :, :]

        x = self.average_pooling_2d(x)

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




class PPOAgent:
    def __init__(self, env, action_dim):
        self.env = env
        self.actor = ActorNetwork(action_dim)
        self.critic = CriticNetwork()
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.95,
            staircase=True)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.gamma = 0.5
        self.lmbda = 0.95
        self.clip_epsilon = 0.3
        self.epochs = 5
        self.batch_size = 32
        self.memory = []


    def remember(self, state, action, reward, next_state, done):
        state = np.reshape(state, (200, 120, 1))
        if done:
            next_state = np.zeros_like(state)
        else:
            next_state = np.reshape(next_state, (200, 120, 1))
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        mean, std = self.actor(state)
        normal_distribution = tfp.distributions.Normal(loc=mean, scale=std)
        action = normal_distribution.sample()
        return action

    def value(self, state):
        state = np.expand_dims(state, axis=0)
        return self.critic(state)

    def sample_memory(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def compute_advantages(self, states, rewards, next_states, dones):
        values = self.critic(states)
        next_values = self.critic(next_states)
        td_errors = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = np.zeros_like(td_errors)
        running_advantage = 0
        # calculates the advantages using the Generalized Advantage Estimation (GAE) method,
        # which is a technique that reduces the variance in advantage estimates
        for t in reversed(range(len(td_errors))):
            running_advantage = td_errors[t] + self.gamma * self.lmbda * running_advantage * (1 - dones[t])
            advantages[t] = running_advantage

        # Normalize the advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        target_values = advantages + values
        return advantages, target_values


    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.sample_memory()
        advantages, target_values = self.compute_advantages(states, rewards, next_states, dones)

        for _ in range(self.epochs):
            for idx in range(0, len(states), self.batch_size):
                batch_indices = list(range(idx, min(idx + self.batch_size, len(states))))
                state_batch = states[batch_indices]
                action_batch = actions[batch_indices]
                advantage_batch = advantages[batch_indices]
                batch_indices_tensor = tf.convert_to_tensor(batch_indices, dtype=tf.int32)
                target_value_batch = tf.gather(target_values, batch_indices_tensor)

                with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                    mean, std = self.actor(state_batch)
                    normal_distribution = tfp.distributions.Normal(loc=mean, scale=std)
                    log_probs = normal_distribution.log_prob(action_batch)
                    selected_probs = tf.exp(log_probs)

                    new_mean, new_std = self.actor(state_batch)
                    new_normal_distribution = tfp.distributions.Normal(loc=new_mean, scale=new_std)
                    new_log_probs = new_normal_distribution.log_prob(action_batch)
                    new_selected_probs = tf.exp(new_log_probs)

                    # Calculate the ratio and clip it
                    ratio = new_selected_probs / selected_probs
                    clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

                    # Calculate the actor loss
                    actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_batch, clipped_ratio * advantage_batch))
                    print(f"actor loss: { actor_loss}")
                    # Calculate the critic loss
                    critic_values = self.critic(state_batch)
                    critic_loss = tf.reduce_mean(tf.square(target_value_batch - critic_values))
                    print(f"critic loss: {critic_loss}")


                # Compute gradients and apply them to the actor and critic networks
                actor_gradients = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
                critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)

                self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        self.memory = []


def main():
    env = GetEnv()

    # Set the hyperparameters
    n_episodes = 1000
    n_timesteps = 100
    update_timestep = 256
    random_seed = None

    if random_seed:
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)

    state_dim = (200, 120, 1)
    action_dim = 1
    ppo = PPOAgent(env, action_dim) # state_dim
    if os.path.exists(file_memory):
        with open(file_memory, "rb") as f:
            ppo.memory = pickle.load(f)

    checkpoint_path = "./checkpoints/ppo_checkpoint"
    ckpt = tf.train.Checkpoint(actor=ppo.actor, critic=ppo.critic)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    running_reward = 0
    timestep = 0

    score = []
    for episode in range(1, n_episodes + 1):
        state = env.reset(is_show=False)
        state = state

        for t in range(n_timesteps):
            timestep += 1

            action = ppo.act(state)
            next_state, reward, done = env.touch_in_step(action)
            ppo.remember(state, action, reward, next_state, done)
            state = next_state
            print(f"update_timestep:{update_timestep}")
            print(f"timestep:{timestep}")
            if timestep % update_timestep == 0:
                ppo.replay()
                timestep = 0
            if done:
                score.append(t)
                break

        running_reward += t
        print('Episode: {}, Timesteps: {}, Running Reward: {}'.format(episode, timestep, running_reward))

        if episode % 10 == 0 and running_reward != 0:
            save_data = (score, episode, running_reward)
            pickle.dump(save_data, open(file_score, 'wb'))
            pickle.dump(ppo.memory, open(file_memory, 'wb'))

            ckpt_save_path = ckpt_manager.save()
            #print("Use All time: {}".format(time.time() - all_time))

if __name__ == '__main__':
    main()


