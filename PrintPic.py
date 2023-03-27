from filecmp import cmp
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import tensorflow as tf
from DDPG import Agent
from pylab import *

sns.set()
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

file_memory = "./data/ddpg_mem.p"
file_score = "./data/ddpg_score.p"


agent = Agent("")

if os.path.exists(file_memory):
    with open(file_memory, "rb") as f:
        agent.memory = pickle.load(f)

checkpoint_path = "./checkpoints/ddpg_checkpoint"
ckpt = tf.train.Checkpoint(actor=agent.actor, actor_target=agent.actor_target, critic=agent.critic, critic_target=agent.critic_target)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

memory = []
if os.path.exists(file_memory):
    with open(file_memory, "rb") as f:
        memory = pickle.load(f)

state_memory = []
for i in memory:
    state_memory.append(i[0])
state_memory = state_memory[-100:]

f, axes = plt.subplots(4,8, figsize=(40,20))

for i in range(4):
    for j in range(8):
        if i % 2 == 0:
            state_index = np.random.choice(len(state_memory))
            pic = state_memory[state_index][0, :, :, 0]
            axes[i][j].imshow(pic, cmap='gray')
            
            action_space = np.linspace(-1,-0,100)
            reward = np.zeros_like(action_space)
            for k in range(action_space.shape[0]):
                reward[k] = agent.critic_target(state_memory[state_index], tf.reshape(action_space[k], [-1, 1]))
            axes[i+1][j].plot(action_space, reward)

plt.savefig("./pic/灰度图与奖励图.png")