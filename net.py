# -*- coding: utf-8 -*-
'''
@brief: Train a Neural Network to play Connect 4
@author: Gabriel Huang
@date: May 2015

This follows the work of Tesauro "Temporal Difference Learning and TD-Gammon" (1995)
'''

import theano
import theano.tensor as T
import lasagne as nn 
import numpy as np
from board import Board
import policy
from profiler import Profiler


input_rows = 15
input_cols = 15

def board_to_feature(board_mat, cache=None):
    if board_mat.shape != (6,7):
        raise Exception('Board must be of size (6,7)')
    if cache is None:
        feature = np.zeros((1, 3, input_rows, input_cols))
    else:        
        feature = cache
        feature[:] = 0.
    offset_rows = input_rows/2 - board_mat.shape[0]/2
    offset_cols = input_cols/2 - board_mat.shape[0]/2
    feature[0,0,offset_rows:offset_rows+board_mat.shape[0],offset_cols:offset_cols+board_mat.shape[1]] = 1 # Border on first row
    feature[0,1,offset_rows:offset_rows+board_mat.shape[0],offset_cols:offset_cols+board_mat.shape[1]][board_mat==1] = 1 # BLACKS
    feature[0,2,offset_rows:offset_rows+board_mat.shape[0],offset_cols:offset_cols+board_mat.shape[1]][board_mat==2] = 1 # REDS
    return feature


def create_from_shared(shared):
    shape = len(shared.get_value().shape)
    if shape==1:
        return T.vector()
    elif shape==2:
        return T.matrix()
    elif shape==3:
        return T.tensor3()
    elif shape==4:
        return T.tensor4()
    raise Exception('Bad size {}'.format(shared.get_value().shape))
    
# Neural Network Params
nepisodes = 30000
eta = 0.1

# TD(lambda) Params
lmbda = 0.7
gamma = 0.999

#%%    

# INPUT LAYER, dimension as the movie (tensor)
l_in = nn.layers.InputLayer((None, 3, input_rows, input_cols))

# FIRST CONVOLUTIONAL LAYER
l_conv1 = nn.layers.Conv2DLayer(l_in, num_filters=4, filter_size=(2, 2),strides=(1, 1),
                                nonlinearity=nn.nonlinearities.rectify)

# FIRST POOLING LAYER. 
l_pool1 = nn.layers.MaxPool2DLayer(l_conv1, ds=(2, 2))

# SECOND CONVOLUTIONAL LAYER
l_conv2 = nn.layers.Conv2DLayer(l_pool1, num_filters=8, filter_size=(2, 2),strides=(1, 1),
                                nonlinearity=nn.nonlinearities.rectify)

# SECOND POOLING LAYER. Downsample a factor of 2x2
l_pool2 = nn.layers.MaxPool2DLayer(l_conv2, ds=(2, 2))

use_third_layer = True
if use_third_layer:
    # THIRD CONVOLUTIONAL LAYER
    l_conv3 = nn.layers.Conv2DLayer(l_pool2, num_filters=16, filter_size=(2, 2),strides=(1, 1),
                                    nonlinearity=nn.nonlinearities.rectify)
    
    # THIRD POOLING LAYER. Downsample a factor of 2x2
    l_pool3 = nn.layers.MaxPool2DLayer(l_conv3, ds=(2, 2))
else:
    l_pool3 = l_pool2

# Output layer, softmax, --------- Gives scores for black
l_out = nn.layers.DenseLayer(l_pool3, num_units=1, nonlinearity=nn.nonlinearities.tanh)
predict = theano.function([l_in.input_var], l_out.get_output())

# Parameters
params = nn.layers.get_all_params(l_out)
print 'Parameters:\n{}'.format(params) # gives the parameters

# Gradient
grads = nn.updates.get_or_compute_grads(l_out.get_output()[0,0], params)
get_grads = theano.function([l_in.input_var], grads)
board = Board()
inp = board_to_feature(board.board)
example_grads = get_grads(inp)
# this is weird !CAREFUL!
smoothed_grads = [theano.shared(np.array(0.*grad)) for grad in example_grads]
smoothed_grads_updates = [(smoothed_grad, grad+lmbda*smoothed_grad) for smoothed_grad,grad in zip(smoothed_grads, grads)]
weight_deltas = [create_from_shared(weight) for weight in params]
#weight_updates = [(weight, weight+weight_delta) for weight, weight_delta in zip(params, weight_deltas)]
factor = T.scalar()
weight_updates_2 = [(weight, weight + factor * smoothed_grad) for weight, smoothed_grad in zip(params, smoothed_grads)]

# Compile theano functions
update_grads = theano.function([l_in.input_var], l_out.get_output(), updates=smoothed_grads_updates)
#update_weights = theano.function(weight_deltas, weight_deltas, updates=weight_updates)

update_weights_2 = theano.function([factor], factor, updates=weight_updates_2)

#%%
def optimal_action(board, color, possible_actions=None, cache=None):
    '''
    This calls the neural network to determine the best action to take
    '''
    if possible_actions is None:
        possible_actions = board.availCols()
    action_scores = []
    cache2 = board.clone().board
    for action in possible_actions:
        possible_board = board.clone(cache2)
        possible_board.play(color, action)
        possible_inp = board_to_feature(possible_board.board, cache)
        action_scores.append((action, predict(possible_inp)))
    action_scores = sorted(action_scores, key=lambda (action,scores): scores[0]
        if color==Board.BLACK else -scores[0], reverse=True)
    best_action = action_scores[0][0]
    return best_action
    
    
class ConvNetPolicy(policy.Policy) :
    def take_action(self, color, board):
        return optimal_action(board, color)
        
    
#%%
board = Board()
inp = board_to_feature(board.board)

#%% Learn Optimal Value
wins = []
profiler = Profiler()
cache = np.zeros((1, 3, input_rows, input_cols))
for episode in xrange(nepisodes):
    if episode%10 == 0: print 'Episode {}/{}'.format(episode+1, nepisodes)
    # Evaluate against Random every once in a while:
    profiler.tic('evaluation')
    if episode % 500 == 0:
        print 'Episode {}/{}'.format(episode+1, nepisodes)
        eval_games = 1000
        convnet_policy = ConvNetPolicy()
        random_policy = policy.RandomPolicy()
        print 'Evaluating ConvnetPolicy against RandomPolicy for {} rounds'.format(eval_games)
        stats = policy.compete_fair(Board(), convnet_policy, random_policy, eval_games)
        print '{} wins, {} draws, {} losses for ConvnetPolicy when playing first'.format(*stats[0])
        print '{} wins, {} draws, {} losses for ConvnetPolicy when playing second'.format(*stats[1])       
        wins.append((episode, stats[0][0], stats[1][0]))
    profiler.toc()        
        
    board = Board()
   
    end_of_game = False
    y_last = 0
    for ply in xrange(7*6):
        color = Board.BLACK if ply%2==0 else Board.RED
        
        # Is there a winner?
        winner = board.getWinner()
        
        # What can we play?
        possible_actions = board.availCols()
        
        # Compute target
        if winner != Board.EMPTY or not possible_actions: # END OF EPISODE
            if winner == Board.EMPTY:
                y_target = 0.
            elif winner == Board.BLACK:
                y_target = 1.
            else:
                y_target = -1.
            end_of_game = True
        else:
            inp = board_to_feature(board.board)
            y_target = predict(inp)[0,0]
        
        # Update smoothed gradients and weights
        profiler.tic('update_grads')
        update_grads(inp)
        profiler.toc()
        if ply>0:
            profiler.tic('update_weights')
            update_weights_2(eta * (y_target - y_last))
            profiler.toc()
            
        # Finish this episode
        if end_of_game:
            break
        
        # Find and play best value for current player color/(ply%2)
        profiler.tic('find_best_action')
        best_action = optimal_action(board, color, possible_actions, cache)
        board.play(color, best_action)
        profiler.toc()
        
        # Backup last value
        y_last = y_target
        
print profiler        
        
#%% Self-Play optimal strategy
board = Board()
for ply in xrange(7*6):
    color = Board.BLACK if ply%2==0 else Board.RED
    winner = board.getWinner()
    possible_actions = board.availCols()
    if winner != Board.EMPTY or not possible_actions: # END OF EPISODE
        break
    
    # Find and play best value for current player color/(ply%2)
    best_action = optimal_action(board, color)
    board.play(color, best_action)
    print '\n{} plays {}'.format(board.to_string(color), best_action)
    print board
print 'Winner is {} after {} moves'.format(board.to_string(color), ply)

#%% Evaluate
eval_games = 10
convnet_policy = ConvNetPolicy()
random_policy = policy.RandomPolicy()
print 'Evaluating ConvnetPolicy against RandomPolicy for {} rounds'.format(eval_games)
stats = policy.compete_fair(Board(), convnet_policy, random_policy, eval_games)
print '{} wins, {} draws, {} losses for ConvnetPolicy when playing first'.format(*stats[0])
print '{} wins, {} draws, {} losses for ConvnetPolicy when playing second'.format(*stats[1])