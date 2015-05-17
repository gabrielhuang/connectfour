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

# Input size
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


#%% Network Architecture

l_in = nn.layers.InputLayer((None, 3, input_rows, input_cols))

l_conv1 = nn.layers.Conv2DLayer(l_in, num_filters=64, filter_size=(4, 4),strides=(1, 1), nonlinearity=nn.nonlinearities.rectify)
l_pool1 = nn.layers.MaxPool2DLayer(l_conv1, ds=(2, 2))



#l_conv2 = nn.layers.Conv2DLayer(l_pool1, num_filters=16, filter_size=(2, 2),strides=(1, 1), nonlinearity=nn.nonlinearities.rectify)
#l_pool2 = nn.layers.MaxPool2DLayer(l_conv2, ds=(2, 2))

#l_conv3 = nn.layers.Conv2DLayer(l_pool2, num_filters=32, filter_size=(3, 3),strides=(1, 1), nonlinearity=nn.nonlinearities.rectify)
#l_pool3 = l_conv3

# MLP alternative
l_hidden = nn.layers.DenseLayer(l_pool1, num_units=400, nonlinearity=nn.nonlinearities.sigmoid)
l_pool3 = l_hidden

l_out = nn.layers.DenseLayer(l_pool3, num_units=4, nonlinearity=nn.nonlinearities.sigmoid)
objective = nn.objectives.Objective(l_out)
cost_var = objective.get_loss()

params = nn.layers.get_all_params(l_out)
print 'Params {}'.format(params)

eta_var = T.scalar()
#updates = nn.updates.sgd(cost_var, params, learning_rate=eta_var)
updates = nn.updates.nesterov_momentum(cost_var, params, learning_rate=eta_var)

predict = theano.function([l_in.input_var], l_out.get_output())
train = theano.function([l_in.input_var, objective.target_var, eta_var], cost_var, updates=updates)
get_cost = theano.function([l_in.input_var, objective.target_var], [cost_var, l_out.get_output()])
#%%

'''
Prepare Dataset
Variable to predict:
y[0] = game over
y[1] = black wins
y[2] = red wins
'''
def create_train_set(ntrain):
    X_train = np.zeros((ntrain, 3, input_rows, input_cols), dtype=np.float32)
    y_train = np.zeros((ntrain, 4), dtype=np.float32)
    board = Board()
    for i in xrange(ntrain):
        board.randomize()
        winner = board.getWinner()
        is_filled = not bool(board.availCols())
        X_train[i] = board_to_feature(board.board)
        y_train[i][0] = 1. if winner==Board.EMPTY else 0.
        y_train[i][1] = 1. if winner|Board.BLACK else 0.
        y_train[i][2] = 1. if winner|Board.RED else 0.
        y_train[i][3] = 1. if is_filled else 0.
        #print 'Filled {} Winner {}'.format(is_filled, winner)
    return X_train,y_train
    


#%%
print 'Go'
epochs = 10000
eta = 0.1
train_costs = []
test_costs = []
test_accuracy = []
batch_size = 20
for epoch in xrange(epochs):
    # REsample trainset
    train_size = 1000
    X_train, y_train = create_train_set(train_size) 
    X_test, y_test = create_train_set(1000) 
    
    batch_costs = []
    for i in xrange(0, train_size//batch_size, batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        # Train the whole batch
        batch_cost = train(X_train, y_train, eta)
        batch_costs.append(batch_cost)
    
    train_cost = np.mean(batch_costs)
    test_cost, y_pred = get_cost(X_test, y_test)
    accuracy = ((y_pred>0.5)==y_test).all(axis=1).mean()
    
    test_costs.append(test_cost)
    test_accuracy.append(accuracy)
    
    print 'Epoch {}'.format(epoch)
    print 'Train cost {}'.format(train_cost)
    print 'Test cost {}'.format(test_cost)
    print 'Test accuracy {}'.format(accuracy)