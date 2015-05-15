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

def board_to_feature(board_mat):
    if board_mat.shape != (6,7):
        raise Exception('Board must be of size (6,7)')
    feature = np.zeros((3, 7, 7))
    feature[0,0,:] = 1 # Border on first row
    feature[1,1:,:][board_mat==1] = 1 # BLACKS
    feature[2,1:,:][board_mat==2] = 1 # REDS
    return feature.reshape((1,3,7,7))

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
nepisodes = 1000
eta = 0.1

# TD(lambda) Params
lmbda = 0.7
gamma = 0.999

#%%    

# INPUT LAYER, dimension as the movie (tensor)
l_in = nn.layers.InputLayer((None, 3, 7, 7))

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

# Output layer, softmax, --------- Gives scores for black
l_out = nn.layers.DenseLayer(l_pool2, num_units=1, nonlinearity=nn.nonlinearities.tanh)
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
weight_updates = [(weight, weight+weight_delta) for weight, weight_delta in zip(params, weight_deltas)]

# Compile theano functions
update_grads = theano.function([l_in.input_var], l_out.get_output(), updates=smoothed_grads_updates)
update_weights = theano.function(weight_deltas, weight_deltas, updates=weight_updates)


def optimal_action(board, color):
    '''
    This calls the neural network to determine the best action to take
    '''
    action_scores = []
    for action in possible_actions:
        possible_board = board.clone()
        possible_board.play(color, action)
        possible_inp = board_to_feature(possible_board.board)
        action_scores.append((action, predict(possible_inp)))
    action_scores = sorted(action_scores, key=lambda (action,scores): scores[0]
        if color==Board.BLACK else -scores[0], reverse=True)
    best_action = action_scores[0][0]
    return best_action
    
#%%
board = Board()
inp = board_to_feature(board.board)
print predict(inp)

#%% Learn Optimal Value
for episode in xrange(nepisodes):
    print 'Episode {}/{}'.format(episode+1, nepisodes)
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
            y_target = np.array([winner]) # (BLACK SCORE, RED SCORE)
            end_of_game = True
        else:
            inp = board_to_feature(board.board)
            y_target = predict(inp)
        
        # Update smoothed gradients and weights
        update_grads(inp)
        if ply>0:
            update_weights(*[eta * (y_target-y_last)[0] * smoothed_grad.get_value() for smoothed_grad in smoothed_grads])        
        
        # Finish this episode
        if end_of_game:
            break
        
        # Find and play best value for current player color/(ply%2)
        best_action = optimal_action(board, color)
        board.play(color, best_action)
        
        # Backup last value
        y_last = y_target
        
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