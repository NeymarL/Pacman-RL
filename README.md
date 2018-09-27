# Pacman-RL

Implement some classic reinforcement learning algorithms, test and visualize on Pacman under [OpenAI's Gym](https://gym.openai.com/) environment.

**Note: This project is still under construction and at the very begining stage!**

## Requirements

* Python 3.6+
* gym
* matplotlib
* keras
* mujoco_py (if you want to save replay)

## Run

For now, just run `python run.py train --controller MC`. The training procedure will start using Monte-Carlo controller.

```
Full usage: run.py [-h] [--controller {MC,Sarsa,Sarsa_lambda,Q_learning}]
              [--render] [--save_replay] [--save_plot] [--show_plot]
              [--num_episodes NUM_EPISODES] [--batch_size BATCH_SIZE]
              [--eva_interval EVA_INTERVAL]
              [--evaluate_episodes EVALUATE_EPISODES]
              [--checkpoints_interval CHECKPOINTS_INTERVAL] [--lr LR]
              [--epsilon EPSILON] [--gamma GAMMA] [--max_workers MAX_WORKERS]
              {train,evaluate}

positional arguments:
  {train,evaluate}      what to do

optional arguments:
  -h, --help            show this help message and exit
  --controller {MC,Sarsa,Sarsa_lambda,Q_learning}
                        choose an algorithm (controller)
  --render              set to render the env when evaluate
  --save_replay         set to save replay
  --save_plot           set to save Q-value plot when evaluate
  --show_plot           set to show Q-value plot when evaluate
  --num_episodes NUM_EPISODES
                        set to run how many episodes
  --batch_size BATCH_SIZE
                        set the batch size
  --eva_interval EVA_INTERVAL
                        set how many episodes evaluate once
  --evaluate_episodes EVALUATE_EPISODES
                        set evaluate how many episodes
  --checkpoints_interval CHECKPOINTS_INTERVAL
                        set how many episodes save the weight once
  --lr LR               set learning rate
  --epsilon EPSILON     set epsilon when use epsilon-greedy
  --gamma GAMMA         set reward decay rate
  --max_workers MAX_WORKERS
                        set max workers to train
```

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

