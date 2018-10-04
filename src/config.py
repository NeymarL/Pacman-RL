import os

from enum import Enum
from logging import getLogger

logger = getLogger(__name__)

def _project_dir():
    d = os.path.dirname
    return d(d(os.path.abspath(__file__)))

class ControllerType(Enum):
    MC = 0
    Sarsa = 1
    Sarsa_lambda = 2
    Q_learning = 3

class Config:
    def __init__(self, controller_type):
        self.controller = ControllerConfig(controller_type)
        self.resource = ResourceConfig()
        self.trainer = TrainerConfig()
        self.render = False
        self.save_replay = False
        self.save_plot = False
        self.show_plot = False
        self.train = True
        self.evaluate = False

        self.resource.set_path(controller_type)
        self.resource.create_directories()

    def set_controller(self, controller_type):
        self.controller = ControllerConfig(controller_type)
        self.resource.set_path(controller_type)
        self.resource.create_directories()

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        logger.info(f"Config.{name} = {value}")

class TrainerConfig:
    def __init__(self):
        self.num_episodes = 10000
        self.batch_size = 50
        self.evaluate_interval = 500
        self.lr = 0.1
        self.evaluate_episodes = 100

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        logger.info(f"TrainerConfig.{name} = {value}")

class ControllerConfig:
    def __init__(self, controller_type):
        self.controller_type = controller_type
        self.epsilon = 0.5
        self.gamma = 0.9
        self.max_workers = 8
        self.lambda_ = 0.5

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        logger.info(f"ControllerConfig.{name} = {value}")

class ResourceConfig:
    def __init__(self):
        self.project_dir = _project_dir()
        self.weight_dir = os.path.join(self.project_dir, 'weights')
        self.graph_dir = os.path.join(self.project_dir, 'graph')
        self.replay_dir = os.path.join(self.project_dir, 'video')
        self.log_path = os.path.join(self.project_dir, 'main.log')

    def set_path(self, controller_type):
        self.weight_path = os.path.join(self.weight_dir, controller_type.name.lower() + '.h5')
        self.graph_dir = os.path.join(self.project_dir, 'graph')
        self.graph_dir = os.path.join(self.graph_dir, controller_type.name.lower())

    def create_directories(self):
        dirs = [self.project_dir, self.weight_dir, self.graph_dir, self.replay_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)
