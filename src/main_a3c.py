import os
import gym
import time
import torch
import numpy as np
import torch.optim as optim
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

from logging import getLogger
from gym.wrappers import Monitor
from torch.autograd import Variable
from torch.multiprocessing import Process

from src.config import Config, ControllerType
from src.a3c import A3CControl, A3CModel

logger = getLogger(__name__)


def main(config: Config):
    # init and load global shared network
    mp.set_start_method('spawn')
    env = gym.make('MsPacman-v0')
    n_channels = env.observation_space.shape[2]
    n_actions = env.action_space.n
    shared_model = A3CModel(n_channels, n_actions)
    # load parameters
    if os.path.exists(config.resource.weight_path):
        logger.info(f"Load weights from {config.resource.weight_path}")
        saved_state = torch.load(config.resource.weight_path)
        shared_model.load_state_dict(saved_state())
    shared_model.share_memory()

    if config.train:
        config.render = False
        config.show_plot = False
        config.save_plot = False
        # train(config, shared_model, 1)
        processes = []
        # start 1 evaluating process
        p = Process(target=evaluate, args=(config, shared_model))
        p.start()
        processes.append(p)
        # start multiple training process
        for rank in range(0, config.controller.max_workers):
            p = Process(target=train, args=(config, shared_model, rank))
            p.start()
            processes.append(p)
            time.sleep(0.1)
        for p in processes:
            p.join()
    elif config.evaluate:
        evaluate(config, shared_model)


def train(config: Config, shared_model, rank):
    ''' One training process
    Create local model and environment,
    update parameters according to A3C algorithm,
    synchronous shared model and local model.

    Args:
        config: configurations of env and model
        shared_model: the PyTorch model of global actor
    '''
    print(f"Process {rank} start training")
    torch.manual_seed(rank)
    env = gym.make('MsPacman-v0')
    controller = A3CControl(env, config)
    env.seed(rank)
    controller.model.train()
    optimizer = optim.RMSprop(shared_model.parameters(), lr=config.trainer.lr)

    t = 1
    done = True
    start_time = time.time()
    while True:
        actions = []
        rewards = []
        log_probs = []
        values = []
        entropies = []
        if done:
            observation = env.reset()
            controller.refresh_state()
            done = False
        # synchronous local model and global model
        controller.model.load_state_dict(shared_model.state_dict())
        # start simulation
        t_start = t
        while not done and t - t_start < config.trainer.t_max:
            try:
                action, value, log_prob, entropy = controller.action(
                    observation, training=True)
            except Exception as e:
                logger.error(e)
            observation, reward, done, _ = env.step(action)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)
            t = t + 1
        # update local parameters
        R = torch.zeros(1, 1)
        if not done:
            _, value, _, _ = controller.action(observation, training=True)
            R = value.data
        optimizer.zero_grad()
        controller.update(rewards, values, log_probs, entropies, R)
        # synchronous update global model and local model
        for param, shared_param in zip(controller.model.parameters(),
                                       shared_model.parameters()):
            shared_param._grad = param.grad
        optimizer.step()
        if done:
            time_diff = time.time() - start_time
            print(f"Process {rank} finish one episode in {time_diff:.2f}s.")
            start_time = time.time()


def evaluate(config: Config, shared_model):
    ''' The evaluation process
    Evaluate the shared model periodically.

    Args:
        config: configurations of env and model
        shared_model: the PyTorch model of global actor
    '''
    print(f"Evaluation process starts")
    env = gym.make('MsPacman-v0')
    if config.render and config.save_replay:
        env = Monitor(env, config.resource.replay_dir, force=True)
    controller = A3CControl(env, config)
    max_reward = 0
    rewards = []
    evaluate_episodes = config.trainer.evaluate_episodes
    while True:
        start_time = time.time()
        # synchronous local model and global model
        controller.model.load_state_dict(shared_model.state_dict())
        controller.model.eval()
        i, mean_reward = 0, 0
        while i < evaluate_episodes:
            mean_reward += _evaluate(controller, env, config, i)
            i += 1
        mean_reward /= evaluate_episodes
        rewards.append(mean_reward)
        _plot_learning_curve(rewards, config)
        cur_time = time.strftime("%Hh %Mm %Ss",
                                 time.gmtime(time.time() - start_time))
        print(f"Evaluate {evaluate_episodes} episodes: "
              f"Time {cur_time}, Mean reward = {mean_reward}")
        if config.train:
            if mean_reward > max_reward:
                max_reward = mean_reward
                print(f"Save weights to {config.resource.weight_path}")
                torch.save(controller.model.state_dict,
                           config.resource.weight_path)
            time.sleep(60)
        else:
            break


def _plot_learning_curve(rewards, config):
    plt.plot(range(len(rewards)), rewards, 'g-')
    plt.title('Learning Curve')
    plt.xlabel('Time')
    plt.ylabel('Mean Reward')
    plt.legend([config.controller.controller_type.name], loc='best')
    plt.savefig(f"{config.resource.graph_dir}/learning_curve.png")


def _evaluate(controller, env, config, i):
    controller.refresh_state()
    observation = env.reset()
    done = False
    total_reward = 0
    values = []
    rewards = []

    if config.show_plot:
        plt.show()
    if config.show_plot or config.save_plot:
        axes = plt.gca()
        axes.set_xlim(0, 1000)
        axes.set_ylim(0, 50)
        axes.set_title('Value and Reward plot')
        axes.set_xlabel('timesteps')
        q_plot, = axes.plot([], [], 'r-')
        r_plot, = axes.plot([], [], 'bx')
        axes.legend(('Value', 'Reward'), loc='best')
        max_val = 0

    while not done:
        if config.render:
            env.render()
        action, value = controller.action(observation, training=False)
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        rewards.append(reward)
        values.append(value)
        if config.show_plot or config.save_plot:
            if max_val < max(reward, value):
                max_val = max(reward, value)
            # draw value function and rewards
            x = range(len(values))
            q_plot.set_xdata(x)
            q_plot.set_ydata(values)
            r_plot.set_xdata(x)
            r_plot.set_ydata(rewards)
            if len(values) >= axes.get_xlim()[1]:
                axes.set_xlim(0, len(values) + 10)
            if max_val >= axes.get_ylim()[1]:
                axes.set_ylim(0, max_val + 10)
            plt.draw()
            if config.show_plot:
                plt.pause(1e-17)
    if config.save_plot:
        plt.savefig(f"{config.resource.graph_dir}/Evaluate_ep{i}.png")
        logger.info(
            f"Total reward = {total_reward} Save sa {config.resource.graph_dir}/Evaluate_ep{i}.png")
    elif config.show_plot:
        logger.info(f"Total reward = {total_reward}")
        plt.close('all')
    return total_reward
