import os
import gym
import copy
import numpy as np
import matplotlib.pyplot as plt

from time import time
from montecarlo import MonteCarloControl
from concurrent.futures import ThreadPoolExecutor, as_completed


def main():
    plt.ion()
    env = gym.make('MsPacman-ram-v0')
    controller = MonteCarloControl(env)
    episodes = 2001
    batch_size = 50
    weight_path = 'mc.h5'
    if os.path.exists(weight_path):
        controller.load(weight_path)
    total_rewards = []
    indexes = []
    i = 0
    while i < episodes:
        if i % 100 == 0:
            total_rewards.append(test(env, controller, i))
            indexes.append(i)
            controller.save(weight_path)

        starttime = time()
        batch_history = []
        batch_rewards = []
        futures = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for j in range(batch_size):
                futures.append(executor.submit(simulation, copy.deepcopy(env), controller))
            for future in as_completed(futures):
                history, rewards = future.result()
                batch_history.append(history)
                batch_rewards.append(rewards)
        i += batch_size
        endtime = time()
        print(f"Episode {i} Observation Finished, {(endtime - starttime):.2f}s")
        starttime = time()
        controller.update_q_value_on_batch_multithreads(batch_history, batch_rewards)
        endtime = time()
        print(f"Episode {i} Learning Finished, {(endtime - starttime):.2f}s")

    plt.plot(indexes, total_rewards, 'g-')
    plt.savefig(f"graph/mc/learn_curve.png")
    print("Learning curve saved as graph/mc/learn_curve.png")

def simulation(env, controller):
    done = False
    observation = env.reset()
    history = []
    rewards = []
    while not done:
        state = np.expand_dims(observation, axis=0)
        action = controller.action(state)
        history.append((observation, action))
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
    return (history, rewards)

def test(env, controller, i):
    observation = env.reset()
    done = False
    total_reward = 0
    qvalues = []
    rewards = []
    plt.show()

    axes = plt.gca()
    axes.set_xlim(0, 1000)
    axes.set_ylim(0, 60)
    q_plot, = axes.plot([], [], 'r-')
    r_plot, = axes.plot([], [], 'bx')
    while not done:
        env.render()
        state = np.expand_dims(observation, axis=0)
        action, Q = controller.action(state, predict=True, return_q=True)
        observation, reward, done, info = env.step(action)
        # print(reward)
        total_reward += reward
        rewards.append(reward)
        qvalues.append(np.max(Q))
        # draw q_value function
        x = range(len(qvalues))
        q_plot.set_xdata(x)
        q_plot.set_ydata(qvalues)
        r_plot.set_xdata(x)
        r_plot.set_ydata(rewards)
        if len(qvalues) > 1000:
            axes.set_xlim(0, len(qvalues) + 10)
        plt.draw()
        plt.pause(1e-17)
    # todo: draw reward
    plt.savefig(f"graph/mc/Ep{i}_Q.png")
    plt.close('all')
    print(f"Episode {i} Game ended! Total reward: {total_reward}")
    return total_reward

if __name__ == '__main__':
    main()
