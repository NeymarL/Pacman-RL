# Pacman-RL

Implement some classic reinforcement learning algorithms, test and visualize on Pacman under [OpenAI's Gym](https://gym.openai.com/) environment.

**Note: This project is still under construction and at the very begining stage!**

## Requirements

* Python 3.6+
* gym
* matplotlib
* keras

## Run

For now, just run `python main.py`. The training procedure will start using Monte-Carlo controller.

## Reinforcement Learning Algorithms

### Monte-Carlo Control

* Policy evaluation
    ![](http://latex.codecogs.com/gif.latex?Q(s_t,a_t)\\leftarrow Q(s_t, a_t) + \\frac{1}{N(s_t, a_t)}(G_t - Q(s_t, a_t)))
    $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \frac{1}{N(s_t, a_t)}(G_t - Q(s_t, a_t))$$
    $$G_t = R_{t + 1} + \gamma R_{t+2} + ... + \gamma^{T-1}R_T$$
* Policy improvement: $\epsilon$-greedy
    $$\pi(a|s) =
\begin{cases} 
\frac{\epsilon}{m} + 1 - \epsilon,  & \mbox{if }a = \argmax_{a\in A}Q(s, a) \\
\epsilon, & \mbox{otherwise}
\end{cases}$$
* Q-value function approximation: 2-layer fully connected layer (input layer and output layer with no hidden layer)


## TODO
* TD(0) control
* TD($\lambda$) control
* Q-learning (or DQN)
* Monte-Carlo policy gradient (REINFORCE)
* Actor-Critic policy gradient

