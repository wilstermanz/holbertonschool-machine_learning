# Deep Q-learning

![](https://miro.medium.com/v2/resize:fit:640/1*GgSiFNufaAWENcpHy5LTgw.gif)

## Learning Objectives

-   What is Deep Q-learning?
-   What is the policy network?
-   What is replay memory?
-   What is the target network?
-   Why must we utilize two separate networks during training?
-   What is keras-rl? How do you use it?

## Requirements

### General
-   Files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
-   Files will be executed with `numpy` (version 1.15), `gym` (version 0.17.2), `keras` (version 2.2.5), and `keras-rl` (version 0.4.2)


## Installing Keras-RL

```
pip install --user keras-rl

```

### Dependencies (that should already be installed)

```
pip install --user keras==2.2.4
pip install --user Pillow
pip install --user h5py

```

## Tasks

### 0. Breakout

Write a python script `train.py` that utilizes `keras`, `keras-rl`, and `gym` to train an agent that can play Atari’s Breakout:

-   Your script should utilize `keras-rl`‘s `DQNAgent`, `SequentialMemory`, and `EpsGreedyQPolicy`
-   Your script should save the final policy network as `policy.h5`

Write a python script `play.py` that can display a game played by the agent trained by `train.py`:

-   Your script should load the policy network saved in `policy.h5`
-   Your agent should use the `GreedyQPolicy`

**Repo:**

-   GitHub repository: `holbertonschool-machine_learning`
-   Directory: `reinforcement_learning/deep_q_learning`
