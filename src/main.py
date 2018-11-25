import os
import gym
import copy
import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger
from gym.wrappers import Monitor
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.util import Buffer
from src.config import Config, ControllerType
from src.montecarlo import MonteCarloControl
from src.sarsa import SarsaControl
from src.sarsa_lambda import SarsaLambdaControl
from src.q_learning import QlearningControl
from src.reinforce import ReinforceControl
from src.actor_critic import ActorCriticControl


logger = getLogger(__name__)


def main(config: Config):
    env = gym.make('MsPacman-ram-v0')
    if config.render and config.save_replay:
        env = Monitor(env, config.resource.replay_dir, force=True)

    controller_dict = {
        ControllerType.MC: MonteCarloControl,
        ControllerType.Sarsa: SarsaControl,
        ControllerType.Sarsa_lambda: SarsaLambdaControl,
        ControllerType.Q_learning: QlearningControl,
        ControllerType.REINFORCE: ReinforceControl,
        ControllerType.ActorCritic: ActorCriticControl
    }

    try:
        controller = controller_dict[config.controller.controller_type](
            env, config)
    except KeyError:
        raise NotImplementedError

    weight_path = config.resource.weight_path
    if os.path.exists(weight_path) or \
            config.controller.controller_type == ControllerType.REINFORCE or \
            config.controller.controller_type == ControllerType.ActorCritic:
        controller.load(weight_path)
    else:
        controller.save(weight_path)

    if config.train:
        config.render = False
        config.show_plot = False
        config.save_plot = False
        train(config, env, controller)
    elif config.evaluate:
        if not config.render and not config.show_plot and not config.save_plot:
            evaluate_parallel(config, env, controller)
        else:
            evaluate(config, env, controller)
    else:
        # do nothing
        pass


def train(config, env, controller):
    episodes = config.trainer.num_episodes
    batch_size = config.trainer.batch_size
    total_rewards = []
    indexes = []
    i = 0
    fail_count = 0
    last_reward = 0
    while i < episodes:
        if i % config.trainer.evaluate_interval == 0:
            current_reward = evaluate_parallel(config, env, controller)
            if len(total_rewards) > 0 and (current_reward > max(total_rewards)):
                controller.save(config.resource.weight_path)
            if len(total_rewards) == 0 or current_reward >= max(total_rewards):
                total_rewards.append(current_reward)
                indexes.append(i)
            if current_reward < last_reward:
                fail_count += 1
                if fail_count >= 5:
                    break
            last_reward = current_reward

        starttime = time()
        batch_buffers = []
        futures = []
        with ThreadPoolExecutor(max_workers=config.controller.max_workers) as executor:
            for _ in range(batch_size):
                futures.append(executor.submit(simulation, copy.deepcopy(
                    env), controller, config.controller.epsilon))
            for future in as_completed(futures):
                buf = future.result()
                batch_buffers.append(buf)
        i += batch_size
        if config.controller.controller_type != ControllerType.REINFORCE:
            # epsilon decay
            config.controller.epsilon *= 0.999 ** batch_size
        endtime = time()
        logger.info(
            f"Episode {i} Observing Finished, {(endtime - starttime):.2f}s")
        starttime = time()
        controller.train(batch_buffers, i)
        endtime = time()
        logger.info(
            f"Episode {i} Learning Finished, {(endtime - starttime):.2f}s")

    plt.plot(indexes, total_rewards, 'g-')
    plt.title('Learning Curve')
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    plt.legend((config.controller.controller_type.name), loc='best')
    plt.savefig(f"{config.resource.graph_dir}/learn_curve.png")
    logger.info(
        f"Learning curve saved as {config.resource.graph_dir}/learn_curve.png")


def simulation(env, controller, epsilon):
    done = False
    observation = env.reset()
    buf = Buffer()
    while not done:
        state = np.expand_dims(observation, axis=0)
        if type(controller) == ControllerType.PPO:
            (action, logp), v = controller.action(
                state, return_q=True)
        else:
            action = controller.action(state, epsilon=epsilon)
            logp = None
            v = [None]
        new_ob, reward, done, _ = env.step(action)
        buf.add(observation, action, reward, v[0], logp)
        observation = new_ob
    return buf


def evaluate(config, env, controller):
    logger.info("Evaluating...")
    mean_reward = 0
    i = 0
    while i < config.trainer.evaluate_episodes:
        mean_reward += eval(controller, env, config, i)
        i += 1
    mean_reward /= config.trainer.evaluate_episodes
    logger.info(f"Mean reward = {mean_reward}")
    return mean_reward


def evaluate_parallel(config, env, controller):
    logger.info("Evaluating...")
    mean_reward = 0
    i = 0
    futures = []
    with ThreadPoolExecutor(max_workers=config.controller.max_workers) as executor:
        while i < config.trainer.evaluate_episodes:
            futures.append(executor.submit(
                eval, controller, copy.deepcopy(env), config, i))
            i += 1
        for future in as_completed(futures):
            mean_reward += future.result()
    mean_reward /= config.trainer.evaluate_episodes
    logger.info(f"Mean reward = {mean_reward}")
    return mean_reward


def eval(controller, env, config, i):
    observation = env.reset()
    done = False
    total_reward = 0
    qvalues = []
    rewards = []

    if config.show_plot:
        plt.show()
    if config.show_plot or config.save_plot:
        axes = plt.gca()
        axes.set_xlim(0, 1000)
        axes.set_ylim(0, 50)
        axes.set_title('Q-value and Reward plot')
        axes.set_xlabel('timesteps')
        q_plot, = axes.plot([], [], 'r-')
        r_plot, = axes.plot([], [], 'bx')
        axes.legend(('Q-value', 'Reward'), loc='best')
        max_val = 0

    while not done:
        if config.render:
            env.render()
        state = np.expand_dims(observation, axis=0)
        if config.controller.controller_type == ControllerType.REINFORCE:
            action = controller.action(state)
            Q = 0
        elif config.controller.controller_type == ControllerType.PPO:
            (action, _), Q = controller.action(state, return_q=True)
        else:
            action, Q = controller.action(state, predict=True, return_q=True)
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        rewards.append(reward)
        maxQ = np.max(Q)
        qvalues.append(maxQ)
        if config.show_plot or config.save_plot:
            if max_val < max(reward, maxQ):
                max_val = max(reward, maxQ)
            # draw q_value function and rewards
            x = range(len(qvalues))
            q_plot.set_xdata(x)
            q_plot.set_ydata(qvalues)
            r_plot.set_xdata(x)
            r_plot.set_ydata(rewards)
            if len(qvalues) >= axes.get_xlim()[1]:
                axes.set_xlim(0, len(qvalues) + 10)
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
