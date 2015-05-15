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
    def take_action(self, color, board):
        '''
        Return an action (integer) given the current board
        '''
        pass


class RandomPolicy:
    '''
    Returns a random play
    '''
    def take_action(self, color, board):
        avail_cols = board.availCols()
        return np.random.choice(avail_cols)

"""
class EpsilonGreedyPolicy:
    '''
    Takes another policy and makes it epsilon-greedy
    '''
    def __init__(self, color, other_policy, epsilon):
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
            return np.random.choice([action for action,score in scores[:idx]])
        else: # No action known for this state
            return self.random_policy.take_action(board)
"""

def compete_one_game(board_prototype, policy_black, policy_red, verbose=False):
    '''
    Make policy_black and policy_red compete against each other for one game
    policy_black always starts and is BLACK
    '''
    board = board_prototype.clone()
    for i in xrange(board.nrows()*board.ncols()):
        color = Board.BLACK if i%2==0 else Board.RED    
        if verbose: print "\nNext: move {}. {}'s turn!".format(i, board.to_string(color))
        if not board.availCols(): break # Nobody can play
           
        # Get new action following policy
        action = (policy_black if color == Board.BLACK else policy_red).take_action(color, board)
        
        # Play action
        winner = board.play(color, action)
        
        if verbose: print board
        if winner != Board.EMPTY: break # We have a winner 
    if verbose: print "\n{} ({}) wins!\n".format(board.to_string(winner), winner)
    return winner
    
    
def compete(board_prototype, policy_black, policy_red, ngames):
    '''
    Make policy_black and policy_red compete against each other for ngames
    policy_black always starts and is BLACK
    
    Return:
    Fraction of wins, draws, and losses
    '''
    wins = 0
    losses = 0
    draws = 0
    for game in xrange(ngames):
        winner = compete_one_game(board_prototype, policy_black, policy_red)
        if winner == Board.BLACK:
            wins += 1
        elif winner == Board.RED:
            losses += 1
        else:
            draws += 1
    return wins/float(ngames),draws/float(ngames),losses/float(ngames)


def compete_fair(board_prototype, policy1, policy2, ngames):
    '''
    Make policy1 and policy2 compete against each other for ngames games A) and B)
    A) The first half of the games policy1 starts and is BLACK
    B) The second half policy2 starts and is BLACK)
    
    Return:
    (fraction of policy1 wins in A, fraction of policy2 wins in B)    
    '''
    a_win,a_draw,a_loss = compete(board_prototype, policy1, policy2, ngames//2)
    b_loss,b_draw,b_win = compete(board_prototype, policy2, policy1, ngames//2)
    return (a_win,a_draw,a_loss),(b_win,b_draw,b_loss)
    

if __name__=='__main__':
    # Run a sample game. Both sides here just play randomly.
    board_prototype = Board(rows=6, cols=7)
    policy1 = RandomPolicy()
    policy2 = RandomPolicy()
    compete_one_game(board_prototype, policy1, policy2, verbose=True)
    
    ngames = 1000
    stats = compete_fair(board_prototype, policy1, policy2, ngames)
    print 'By playing against itself {} rounds, RandomPolicy achieves:'.format(ngames)
    print '{} wins, {} draws, {} losses when playing first'.format(*stats[0])
    print '{} wins, {} draws, {} losses when playing second'.format(*stats[1])