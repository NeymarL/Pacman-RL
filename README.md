# Pacman-RL

Implement some classic reinforcement learning algorithms, test and visualize on Pacman under [OpenAI's Gym](https://gym.openai.com/) environment.

**Note: This project is still under construction and at the very begining stage!**

## Requirements

* Python 3.6+
* gym
* matplotlib
* keras

## Run

For now, just run `python run.py train --controller MC`. The training procedure will start using Monte-Carlo controller.

## Reinforcement Learning Algorithms

### Monte-Carlo Control


* Policy evaluation
    * ![](http://latex.codecogs.com/gif.latex?Q%28s_t%2C%20a_t%29%20%5Cleftarrow%20Q%28s_t%2C%20a_t%29%20&plus;%20%5Cfrac%7B1%7D%7BN%28s_t%2C%20a_t%29%7D%28G_t%20-%20Q%28s_t%2C%20a_t%29%29)
    * ![](http://latex.codecogs.com/gif.latex?G_t%20%3D%20R_%7Bt%20&plus;%201%7D%20&plus;%20%5Cgamma%20R_%7Bt&plus;2%7D%20&plus;%20...%20&plus;%20%5Cgamma%5E%7BT-1%7DR_T)

* Policy improvement: 𝜀-greedy
* Q-value function approximation: 2-layer fully connected layer (input layer and output layer with no hidden layer)


## TODO
* Sarsa(0)
* Sarsa(ƛ)
* Q-learning (or DQN)
* Monte-Carlo policy gradient (REINFORCE)
* Actor-Critic policy gradient

