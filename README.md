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

* Run `python run.py --controller MC train` for training using Monte-Carlo control. The weight file will be saved as `weights/mc.h5`.
* Run `python run.py --controller MC --render --show_plot --evaluate_episodes 10 evaluate` for evaluation using Monte-Carlo control. It will render the Pacman environment and show the dynamic Q-value and reward plot at the same time.

```
Full usage: run.py [-h] [--controller {MC,Sarsa,Sarsa_lambda,Q_learning}]
              [--render] [--save_replay] [--save_plot] [--show_plot]
              [--num_episodes NUM_EPISODES] [--batch_size BATCH_SIZE]
              [--eva_interval EVA_INTERVAL]
              [--evaluate_episodes EVALUATE_EPISODES] [--lr LR]
              [--epsilon EPSILON] [--gamma GAMMA] [--lam LAM]
              [--max_workers MAX_WORKERS]
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
  --lr LR               set learning rate
  --epsilon EPSILON     set epsilon when use epsilon-greedy
  --gamma GAMMA         set reward decay rate
  --lam LAM             set lambda if use sarsa(lambda) algorithm
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

### Sarsa(0)

* Policy evaluation
    * ![](http://latex.codecogs.com/gif.latex?Q%28s%2Ca%29%5Cleftarrow%20Q%28s%2Ca%29&plus;%5Calpha%28R&plus;%5Cgamma%20Q%28s%27%2Ca%27%29-Q%28s%2Ca%29%29)
* Policy improvement: 𝜀-greedy
* Q-value function approximation: 2-layer fully connected layer (input layer and output layer with no hidden layer)

### Sarsa(𝝀)

* Policy evaluation
    * ![](http://latex.codecogs.com/gif.latex?Q%28s%2C%20a%29%20%5Cleftarrow%20Q%28s%2C%20a%29%20&plus;%20%5Calpha%28q_t%5E%5Clambda-Q%28s%2Ca%29%29)
    * ![](http://latex.codecogs.com/gif.latex?q_t%5E%5Clambda%3D%281-%5Clambda%29%5Csum_%7Bn%3D1%7D%5E%5Cinfty%20%5Clambda%5E%7Bn-1%7Dq_t%5E%7B%28n%29%7D)
    * ![](http://latex.codecogs.com/gif.latex?q_t%5E%7B%28n%29%7D%3DR_%7Bt&plus;1%7D%20&plus;%20%5Cgamma%20R_%7Bt&plus;2%7D%20&plus;%20...%20&plus;%20%5Cgamma%5E%7Bn-1%7D%20R_%7Bt&plus;n%7D&plus;%5Cgamma%5En%20Q%28s_%7Bt&plus;n%7D%2C%20a_%7Bt&plus;n%7D%29)
* Policy improvement: 𝜀-greedy
* Q-value function approximation: 2-layer fully connected layer (input layer and output layer with no hidden layer)

## TODO
* Q-learning
* Monte-Carlo policy gradient (REINFORCE)
* Actor-Critic policy gradient
