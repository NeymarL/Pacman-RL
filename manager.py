import argparse

from logging import getLogger

from src.logger import setup_logger
from src.config import Config, ControllerType

logger = getLogger(__name__)
CMD_LIST = ['train', 'evaluate']


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="what to do", choices=CMD_LIST)
    parser.add_argument("--controller", help="choose an algorithm (controller)",
                        choices=[name for name, m in ControllerType.__members__.items()])
    parser.add_argument(
        "--render", help="set to render the env when evaluate", action="store_true")
    parser.add_argument(
        "--save_replay", help="set to save replay", action="store_true")
    parser.add_argument(
        "--save_plot", help="set to save Q-value plot when evaluate", action="store_true")
    parser.add_argument(
        "--show_plot", help="set to show Q-value plot when evaluate", action="store_true")
    parser.add_argument(
        "--num_episodes", help="set to run how many episodes", default=10000, type=int)
    parser.add_argument(
        "--batch_size", help="set the batch size", default=50, type=int)
    parser.add_argument(
        "--eva_interval", help="set how many episodes evaluate once", default=500, type=int)
    parser.add_argument("--evaluate_episodes",
                        help="set evaluate how many episodes", default=100, type=int)
    parser.add_argument("--lr", help="set learning rate",
                        default=0.0001, type=float)
    parser.add_argument(
        "--epsilon", help="set epsilon when use epsilon-greedy", default=0.5, type=float)
    parser.add_argument(
        "--gamma", help="set reward decay rate", default=0.9, type=float)
    parser.add_argument(
        "--lam", help="set lambda if use sarsa(lambda) algorithm", default=0.5, type=float)
    parser.add_argument(
        "--forward", help="set to use forward-view sarsa(lambda)", action="store_true")
    parser.add_argument(
        "--max_workers", help="set max workers to train", default=8, type=int)
    parser.add_argument(
        "--t_max", help="set simulate how many timesteps until update param", default=5, type=int)
    return parser


def start():
    parser = create_parser()
    args = parser.parse_args()

    config = Config(ControllerType[args.controller])

    if config.controller.controller_type == ControllerType.A3C:
        from src.main_a3c import main
    else:
        from src.main import main

    if args.cmd == 'train':
        config.train = True
        config.evaluate = False
    else:
        config.train = False
        config.evaluate = True
    print("\n===============================================================")
    config.render = args.render
    config.save_replay = args.save_replay
    config.show_plot = args.show_plot
    config.save_plot = args.save_plot
    config.trainer.num_episodes = args.num_episodes
    config.trainer.batch_size = args.batch_size
    config.trainer.evaluate_interval = args.eva_interval
    config.trainer.lr = args.lr
    config.trainer.evaluate_episodes = args.evaluate_episodes
    config.controller.epsilon = args.epsilon
    config.controller.gamma = args.gamma
    config.controller.lambda_ = args.lam
    config.controller.forward = args.forward
    config.controller.max_workers = args.max_workers
    config.trainer.t_max = args.t_max
    print("===============================================================\n")
    main(config)
