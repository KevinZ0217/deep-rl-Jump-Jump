import tensorflow as tf
import numpy as np
from numpy import random
from GetEnv import GetEnv
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

        self.outputs = tf.keras.layers.Dense(action_dim, activation='softmax')

    def call(self, inputs, training=False):
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

        outputs = self.outputs(x)

        return outputs


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

    def call(self, inputs, training=False):
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

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.gamma = 0.99
        self.lmbda = 0.95
        self.clip_epsilon = 0.2
        self.epochs = 10
        self.batch_size = 64
        self.memory = []

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.expand_dims(state, axis=0)
        action_probs = self.actor(state)
        action = np.random.choice(range(action_probs.shape[1]), p=action_probs.numpy().ravel())
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
        for t in reversed(range(len(td_errors))):
            running_advantage = td_errors[t] + self.gamma * self.lmbda * running_advantage * (1 - dones[t])
            advantages[t] = running_advantage
        target_values = advantages + values
        return advantages, target_values

    # Include the previously provided replay function here

    def replay(self):
        states, actions, rewards, next_states, dones = self.sample_memory()
        advantages, target_values = self.compute_advantages(states, rewards, next_states, dones)

        for _ in range(self.epochs):
            for idx in range(0, len(states), self.batch_size):
                batch_indices = range(idx, min(idx + self.batch_size, len(states)))
                state_batch = states[batch_indices]
                action_batch = actions[batch_indices]
                advantage_batch = advantages[batch_indices]
                target_value_batch = target_values[batch_indices]

                with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                    # Calculate the probabilities for the selected actions
                    action_probs = self.actor(state_batch)
                    selected_probs = tf.reduce_sum(action_probs * tf.one_hot(action_batch, action_probs.shape[1]),
                                                   axis=1)

                    # Calculate the new probabilities after updating the actor
                    new_action_probs = self.actor(state_batch)
                    new_selected_probs = tf.reduce_sum(
                        new_action_probs * tf.one_hot(action_batch, new_action_probs.shape[1]), axis=1)

                    # Calculate the ratio and clip it
                    ratio = new_selected_probs / selected_probs
                    clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

                    # Calculate the actor loss
                    actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_batch, clipped_ratio * advantage_batch))

                    # Calculate the critic loss
                    critic_values = self.critic(state_batch)
                    critic_loss = tf.reduce_mean(tf.square(target_value_batch - critic_values))

                # Compute gradients and apply them to the actor and critic networks
                actor_gradients = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
                critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)

                self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        self.memory = []


# import numpy as np
# import tensorflow as tf
# from ppo import PPOAgent


def main():
    env = GetEnv()

    # Set the hyperparameters
    n_episodes = 1000
    n_timesteps = 200
    update_timestep = 2000
    lr = 0.0025
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2
    random_seed = None

    if random_seed:
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)

    state_dim = 200 * 120
    action_dim = 1
    ppo = PPOAgent(state_dim, action_dim)

    running_reward = 0
    timestep = 0

    for episode in range(1, n_episodes + 1):
        state = env.reset(is_show=False)
        state = state.reshape(-1)

        for t in range(n_timesteps):
            timestep += 1

            action = ppo.act(state)
            next_state, reward, done = env.touch_in_step(action)

            if done:
                break

            ppo.remember(state, action, reward, next_state, done)
            state = next_state.reshape(-1)

            if timestep % update_timestep == 0:
                ppo.replay()
                timestep = 0

        running_reward += t
        print('Episode: {}, Timesteps: {}, Running Reward: {}'.format(episode, t, running_reward))

if __name__ == '__main__':
    main()


