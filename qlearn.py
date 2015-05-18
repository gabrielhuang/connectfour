# -*- coding: utf-8 -*-
'''
@brief Q-learning for the game of Connect 4
@author: Gabriel Huang
@date: May 2015

This uses Q-learning similar to 
Mnih et al. 2013 "Playing Atari with Deep Reinforcement Learning"
http://arxiv.org/pdf/1312.5602.pdf
'''

import numpy as np
import theano
import theano.tensor as T
import lasagne as nn
from board import Board
import policy

# Input size
input_rows = 9
input_cols = 9

def board_to_feature(board_mat, cache=None):
    '''
    Transforms the input 6 x 7 board "board_mat", by centering it 
    into a 3 x input_rows x input_cols 3D tensor "feature", 
    where:
        feature[0,i,j] = 1 if board_mat[i,j] == 1 if in centered board else 0
        feature[1,i,j] = 1 if board_mat[i,j] == "BLACK" else 0
        feature[2,i,j] = 1 if board_mat[i,j] == "RED" else 0
    '''
    if board_mat.shape != (6,7):
        raise Exception('Board must be of size (6,7)')
    if cache is None:
        feature = np.zeros((1, 3, input_rows, input_cols), dtype=np.float32)
    else:        
        feature = cache
        feature[:] = 0.
    offset_rows = input_rows/2 - board_mat.shape[0]/2
    offset_cols = input_cols/2 - board_mat.shape[0]/2
    feature[0,0,offset_rows:offset_rows+board_mat.shape[0],offset_cols:offset_cols+board_mat.shape[1]] = 1 # Border on first row
    feature[0,1,offset_rows:offset_rows+board_mat.shape[0],offset_cols:offset_cols+board_mat.shape[1]][board_mat==1] = 1 # BLACKS
    feature[0,2,offset_rows:offset_rows+board_mat.shape[0],offset_cols:offset_cols+board_mat.shape[1]][board_mat==2] = 1 # REDS
    return feature


################################
# Network Architecture
################################

# Input: Board State s
#
# Output: 7 + 7 neurons corresponding to Q(s,a) where:
#   the 7 first neurons correspond to actions a for player BLACK
#   the 7 next neurons correspond to actions a for player RED
#
# Each player has its own Q function, but they share features
l_in = nn.layers.InputLayer((None, 3, input_rows, input_cols))

# Conv + MaxPool
l_conv1 = nn.layers.Conv2DLayer(l_in, num_filters=16, filter_size=(4, 4),strides=(1, 1), nonlinearity=nn.nonlinearities.rectify)
l_pool1 = nn.layers.MaxPool2DLayer(l_conv1, ds=(2, 2))

# Conv + MaxPool
l_conv2 = nn.layers.Conv2DLayer(l_pool1, num_filters=32, filter_size=(2, 2),strides=(1, 1), nonlinearity=nn.nonlinearities.rectify)
l_pool2 = nn.layers.MaxPool2DLayer(l_conv2, ds=(2, 2))

# Optional hidden layer
#l_dense1 = nn.layers.DenseLayer(l_pool2, num_units=256, nonlinearity=nn.nonlinearities.rectify)

# Output layer
l_out = nn.layers.DenseLayer(l_pool2, num_units=14, nonlinearity=nn.nonlinearities.sigmoid)
objective = nn.objectives.Objective(l_out) # MSE
cost_var = objective.get_loss()

params = nn.layers.get_all_params(l_out)
print 'Params {}'.format(params)

eta_var = T.scalar()
updates = nn.updates.nesterov_momentum(cost_var, params, learning_rate=eta_var)

predict = theano.function([l_in.input_var], l_out.get_output())
get_cost = theano.function([l_in.input_var, objective.target_var], [cost_var, l_out.get_output()])

# Backprop on a single output neuron y_j
# example:
#   to backprop on neuron y_j for a single example (X[i],y[i])
#   use select_var[i][j] = 1 and select_var[i][k] for k != j
#   then call train_action(X, y, select_var)
select_var = T.matrix()
l_select = nn.layers.InputLayer((None, 1))
l_select.input_var = (l_out.get_output() * select_var).sum(axis=1).reshape((-1,1))
select_objective = nn.objectives.Objective(l_select) # MSE
select_updates = nn.updates.sgd(select_objective.get_loss(), params, eta_var)
train_action = theano.function([l_in.input_var, select_objective.target_var, select_var, eta_var], 
                               select_objective.get_loss(), updates=select_updates)

def q_best_action(color, board):
    '''
    Given the trained Q-neural-network,
    return the best valid action for player "color"
    i.e., argmax(a') Q(s,a')
    '''
    possible_actions = board.availCols()
    inp = board_to_feature(board.board)
    scores = predict(inp)[0]
    offset = 0 if color == Board.BLACK else 7 # which action value to look at?
    actions = [(scores[offset+action], action) for action in possible_actions]
    actions.sort(reverse=True)
    best_action = actions[0][1]
    return best_action
    
class QPolicy(policy.Policy):
    '''
    Follows policy dictated by Q
    '''
    def take_action(self, color, board):
        return q_best_action(color, board)
        


################################
# Q-Learning
################################
ngames = 5000
experience_size = 1000000
batch_size = 32
gamma = 0.9 # discounting factor
get_epsilon = lambda t: max(0.1, 1-float(t)/(ngames/4)) # probability of random action
get_eta = lambda t: np.float32(0.01) # learning rate

wins = []
experience = []
random_policy = policy.RandomPolicy()
X_batch = np.zeros((batch_size, 3, input_rows, input_cols), dtype=np.float32)
y_batch = np.zeros((batch_size, 1), dtype=np.float32)
select_batch = np.zeros((batch_size, 14), dtype=np.float32) 

for t in xrange(ngames):
    eta = get_eta(t)
    epsilon = get_epsilon(t)
    board = Board()
    for ply in xrange(board.ncols()*board.nrows()):
        color = Board.BLACK if ply%2==0 else Board.RED # who plays this ply    
        if np.random.uniform()<epsilon: # take random action
            action = random_policy.take_action(color, board)        
        else: # follow Q-greedy policy
            action = q_best_action(color, board)
        new_board = board.clone()
        new_board.play(color, action)
        winner = new_board.getWinner()
        
        if winner == color:
            reward = 1.
        elif winner == Board.EMPTY:
            reward = 0.
        else:
            reward = -1.
        
        inp = board_to_feature(board.board)        
        new_inp = board_to_feature(new_board.board)     

        end_of_game = (winner != Board.EMPTY) or (not new_board.availCols())     
    
        experience.append((inp, action, reward, end_of_game, color, new_inp))
        # Sample batch from experience
        batch_idx = np.random.choice(len(experience), batch_size)
        select_batch[:] = 0.
        for i, idx in enumerate(batch_idx):
            s, a, r, eog, col, s2 = experience[idx]
            offset = 0 if col == Board.BLACK else 7
            X_batch[i] = s
            y_batch[i] = r
            select_batch[i][offset+a] = 1.
            if not eog: # not end of game, add discounted reward
                scores = predict(s)[0]
                other_reward = np.max(scores[offset:offset+7])
                y_batch[i] = r + gamma * other_reward
        # Perform gradient descent Q(s,a) ~> r + gamma * max(a') Q(s',a')
        train_action(X_batch, y_batch, select_batch, eta)
        
        board = new_board
        if end_of_game:
            break
    if len(experience)>experience_size:
        experience = experience[-experience_size:]

    if (t+1)%10 == 0:
        print 'Game {}/{} --> {} plies'.format(t+1, ngames, ply+1)
        print 'eta {} epsilon {}'.format(eta, epsilon)
        print 'experience {}, batch {}'.format(len(experience), batch_size)
        # evaluate against RandomPolicy
        stats = policy.compete(Board(), QPolicy(), policy.RandomPolicy(), 100)
        wins.append((t, stats[0]))
        print '-'*32
        print 'wins/draws/losses against random: {}'.format(stats)
        print '-'*32


################################
# Results
################################
#%%
import matplotlib.pyplot as plt
print 'Evolution of wins'
wins_ = np.array(wins)
window = 20
smoothed = np.convolve(wins_[:,1],[1./window]*window, mode='same')
plt.plot(wins_[:,0], wins_[:,1], wins_[:,0], smoothed)

#%%
print 'Now see the network self play against itself'
policy.compete_one_game(Board(), QPolicy(), QPolicy(), verbose=True)

#%%
print 'Now play against random'
policy.compete_one_game(Board(), QPolicy(), policy.RandomPolicy(), verbose=True)