# -*- coding: utf-8 -*-
'''
@brief: Train a Neural Network to play Connect 4
@author: Gabriel Huang
@date: May 2015

This follows the work of Tesauro "Temporal Difference Learning and TD-Gammon" (1995)
'''

from board import Board
from policy import RandomPolicy, QGreedyPolicy, EpsilonGreedyPolicy
import numpy as np

def raw_feature(board):
    return board.board


def tensor_3d(board):
    feature = np.zeros((2, board.board.shape[0], board.board.shape[1]))
    feature[0] = (board.board == Board.BLACK)
    feature[1] = (board.board == Board.RED)
    return feature
    

def get_reward(board, we_are=Board.BLACK):
    '''
    Return reward for being in given state
    By default we are BLACK
    '''
    winner = board.getWinner()
    if winner == we_are:
        return 1
    elif winner == board.EMPTY:
        return 0
    else:
        return -1


def td0_estimate_value(board_prototype, policy, nepisodes, alpha, gamma):
    '''
    Value estimation using TD(0), see:
    http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node61.html
    
    Parameters:
    :board_prototype: an empty board that will generate all initial states
    :policy: class derived from board.Policy
    :nepisodes: episodes to simulate
    '''
    V = {}
    for episode in xrange(nepisodes):
        # Create empty board with right size
        board = board_prototype.clone()
        for i in range(board.ncols()*board.nrows()):
            color = Board.BLACK if i%2 else Board.RED
            
            # s
            old_state = board.to_tuple()
            
            # a
            action = policy.take_action(board)
            
            # play s, get r_t            
            winner = board.play(color, action)
            reward = get_reward(board, we_are=Board.BLACK)
            
            # s'
            new_state = board.to_tuple()         # only immutables can be keys    
            
            # V(s) <- V(s) + alpha * (r_t + gamma * V(s') - V(s))
            V.setdefault(old_state, 0.)
            V[old_state] = V[old_state] + alpha * (reward + gamma * V.get(new_state, 0.) - V[old_state])

                
            if winner != Board.EMPTY:
                break 
    return V


def max_action(q_s, value_if_empty=None):
    '''
    Return max_a q_s[a]
    '''
    if q_s:
        return max(q_s.values())
    else:
        return value_if_empty


def q_learn(board_prototype, nepisodes, alpha, gamma, epsilon):
    '''
    Q-Learning using Epsilon-greedy policy
    http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node65.html
    '''
    global Q
    Q = {}
    for episode in xrange(nepisodes):
        # Create empty board with right size
        board = board_prototype.clone()
        for i in range(board.ncols()*board.nrows()):
            q_greedy_policy = QGreedyPolicy(Q)
            eps_greedy_policy = EpsilonGreedyPolicy(q_greedy_policy, epsilon)
            
            color = Board.BLACK if i%2 else Board.RED
            
            old_state = board.to_tuple()      # s
            
            if color == Board.RED:
                board.flip()
            action = eps_greedy_policy.take_action(board) # a
            winner = board.play(color, action)
            reward = get_reward(board, we_are=Board.BLACK) # r_t
            if color == Board.RED:
                board.flip()            
            
            new_state = board.to_tuple()         # s'
            
            Q.setdefault(old_state, {})
            Q[old_state].setdefault(action, 0.)
            current = Q[old_state][action] # Q(s,a)

            Q.setdefault(new_state, {})
            best = max_action(Q[new_state], value_if_empty=0.) # max_a Q(s',a)
            
            # Q(s,a) <- Q(s,a) + alpha * (r_t + gamma * max_a Q(s',a) - Q(s,a))
            Q[old_state][action] = current + alpha * (reward + gamma * best - current)
            if winner != Board.EMPTY:
                break 
    return Q


################################################
action = 'q'

if __name__=='__main__' and action=='value':
    board_prototype = Board(rows=3, cols=3, to_win=3) # board generator
    
    print 'Learning values for RandomPolicy'
    policy = RandomPolicy()
    nepisodes = 10000
    alpha = 0.05
    gamma = 0.999
    V = td0_estimate_value(board_prototype, policy, nepisodes, alpha, gamma)
    U = sorted(V.items(), key=lambda (a,b):b, reverse=True)
    
    print 'Best Case\n {}'.format('\n'.join(map(str,U[0][0])))
    print 'Worst Case\n {}'.format('\n'.join(map(str,U[-1][0])))
    
    print len(V)
    
if __name__=='__main__' and action=='q':
    board_prototype = Board(rows=3, cols=3, to_win=3) # board generator
    
    print 'Learning values for RandomPolicy'
    policy = RandomPolicy()
    nepisodes = 10000
    alpha = 0.05
    gamma = 0.999
    epsilon = 0.1
    Q = q_learn(board_prototype, nepisodes, alpha, gamma, epsilon)
    print len(Q)
    
    #%%
    board = Board(rows=3, cols=3, to_win=3)
    board.board = np.array([
    [2,0,0],
    [2,0,2],
    [1,0,1]   
    ])
    
    action = QGreedyPolicy(Q).take_action(board)
    
    print 'Best action in \n{}\n --> {}'.format(board, action)