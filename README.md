Q-Learning for Connect-Four
===========================

This implements Q-learning similarly to 
Mnih et al. 2013 "Playing Atari with Deep Reinforcement Learning"
http://arxiv.org/pdf/1312.5602.pdf

Dependencies
------------

- Python 2.7 with numpy/scipy
- Theano
- Lasagne

`board.py`
--------

Implements a Connect-Four game board (6 rows, 7 columns by default)

`policy.py`
---------

Implements a base class and various policies for playing Connect-Four.
A policy takes a game state (a `Board` object) and returns an optimal action (in some sense)

`qlearn.py`
---------

Attempts to train a Convolutional Neural Network (CNN) using Q-learning
to predict`Q(s,a)` the value of taking each action `a` in state `s`
The input is `s` and there is one output neuron for each `a` corresponding to `Q(s,a)`.

`net2.py`
-------

Trains a CNN to predict the winner(s) of Connect-Four and other stuff in a supervised way:
random boards (X) are generated and the answers (y) are computed.
The CNN is then trained using gradient descent on examples (X,y).

