# -*- coding: utf-8 -*-
'''
@brief: Defines policies to be used with the "board" module
@author: Gabriel Huang (modified)
@data: May 2015

A policy takes a board state as input and returns an action to take
'''
import numpy as np
from board import Board


class Policy:
    '''
    Abstract class for Connect 4 policies
    '''
    def take_action(self, board):
        '''
        Return an action (integer) given the current board
        '''
        pass


class RandomPolicy:
    '''
    Returns a random play
    '''
    def take_action(self, board):
        avail_cols = board.availCols()
        return np.random.choice(avail_cols)


class EpsilonGreedyPolicy:
    '''
    Takes another policy and makes it epsilon-greedy
    '''
    def __init__(self, other_policy, epsilon):
        self.other_policy = other_policy
        self.random_policy = RandomPolicy()
        self.epsilon = epsilon
    def take_action(self, board):
        if np.random.uniform()>self.epsilon:
            return self.other_policy.take_action(board)
        else:
            return self.random_policy.take_action(board)
            
            
class QGreedyPolicy:
    '''
    Takes a policy greedy relative to Q(s,a)
    If many a have the same Q-value, sample uniformly
    
    Q['state']['action'] = value or 0. if invalid
    '''
    def __init__(self, q):
        self.q = q
        self.random_policy = RandomPolicy()
    def take_action(self, board):  
        current_state = board.to_tuple()
        if current_state in self.q and self.q[current_state]:
            scores = sorted(self.q[current_state].items(), key=lambda (action,score):score, reverse=True)
            idx = 1
            while idx < len(scores):
                if scores[idx] == scores[0]:
                    break
                else:
                    idx += 1
            return np.random.choice((action for action,score in scores[:idx]))
        else: # No action known for this state
            return self.random_policy.take_action(board)


if __name__=='__main__':
    # Run a sample game. Both sides here just play randomly.
    board = Board(rows=6, cols=7)
    policy = RandomPolicy()
    winner = Board.EMPTY
    for i in xrange(board.nrows()*board.ncols()):
        color = Board.BLACK if i%2 else Board.RED    
        print "\nNext: move {}. {}'s turn!".format(i, board.to_string(color))
        if not board.availCols(): break # Nobody can play
           
        # Get new action following policy
        action = policy.take_action(board)
        
        # Play action
        winner = board.play(color, action)
        
        print board
        if winner != Board.EMPTY: break # We have a winner
          
    # Print result
    print "\n{} ({}) wins!\n".format(board.to_string(winner), winner)
