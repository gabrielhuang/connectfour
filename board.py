# -*- coding: utf-8 -*-
'''
@brief: A board for Connect Four and abstract policies
@author: Evan Chow (created), Gabriel Huang (modified)
@data: May 2015

Currently does not work for diagonal wins (e.g. 5 in 
a row along a diagonal).
'''

import numpy as np


class Board():
    '''
    The idea is that you place your pieces at the top, and they
    drop down to populate the first empty square closest to the bottom.
    
    Convention:
    EMPTY == 0
    BLACK == 1
    RED == 2
    '''
    EMPTY = 0
    BLACK = 1
    RED = 2

    def __init__(self, rows=6, cols=7, to_win=4, cache=None):
        '''
        Init empty board
        '''
        self.to_win = to_win
        if cache is None:
            self.board = np.zeros((rows, cols))
        else:
            self.board = cache
            self.board[:] = 0.

    def clone(self, cache=None):
        '''
        Return HARD copy of current board
        '''
        other = Board(self.nrows(), self.ncols(), self.to_win, cache)
        other.board[:,:] = self.board
        
        return other

    def to_string(self, index):
        '''
        Convert index to corresponding BLACK/RED/EMPTY
        '''
        if index==self.EMPTY:
            return 'EMPTY'
        elif index==self.BLACK:
            return 'BLACK'
        elif index==self.RED:
            return 'RED'
        else:
            raise Exception('Bad value {}'.format(index))

    def flip(self):
        '''
        Swap REDS and BLACKS
        '''
        red_pos = (self.board==self.RED)
        self.board[self.board==self.BLACK] = self.RED
        self.board[red_pos] = self.BLACK

    def nrows(self):
        return self.board.shape[0]
        
    def ncols(self):
        return self.board.shape[1]

    def rows(self):
        return (self.row(i) for i in xrange(self.nrows()))

    def cols(self):
        return (self.col(j) for j in xrange(self.ncols()))

    def col(self, col):
        return self.board[:, col]

    def row(self, row):
        return self.board[row, :]

    def play(self, color, col):
        '''
        Place a piece at a given column.
        
        Parameters:
        :color: player id, either Board.RED or Board.BLACK 
        
        Returns:
        Game status
        '''
        # Iterate bottom up until hit a 0; otherwise invalid move!
        for sx in xrange(self.nrows()-1, -1, -1):
            if self.board[sx, col] == self.EMPTY:
                self.board[sx, col] = color
                return self.getWinner()

         # if hit top of board without empty square
        raise Exception("No more moves in that column!")
        
    def hasEnoguhAligned(self, vector, n_to_win=4):
        '''
        For checking whether the board is solved
        Given a vector, see if it has a winner (n_to_win of same piece in row).
        
        Parameters:
        :n_to_win: number to win (typically 4)
        '''
        last = 0
        for i in xrange(len(vector)):
            if vector[i] != vector[last]:
                last = i
            if i-last+1 >= n_to_win and vector[i] != self.EMPTY:
                return vector[i]
        return self.EMPTY
        
    def getWinner(self):
        '''
        Return:
        Board.RED if RED has won, 
        Board.BLACK if BLACK has won, 
        Board.EMPTY if neither
        '''
        # Check if any of the columns have winners.
        for col in self.cols():
            result = self.hasEnoguhAligned(col, self.to_win)
            if result != self.EMPTY:
                return result
        # Check if any of the rows have winners.
        for row in self.rows():
            result = self.hasEnoguhAligned(row, self.to_win)
            if result != self.EMPTY:
                return result
        # No winners
        return self.EMPTY

    def availCols(self):
        '''
        Return whichever columns are available for more moves.
        '''
        return [i for i in xrange(self.ncols()) if self.board[0,i] == 0]

    def randomize(self):
        '''
        Randomize the board with 0's, 1's, and 2's
        Accounts for gravity (pieces fall to bottom). 
        Does not check if the board is solved.
        '''
        self.board[:] = 0.
        num_pieces = np.random.randint(self.nrows() * self.ncols())
        
        for i in xrange(num_pieces):
            color = self.BLACK if i%2 else self.RED          
            avail_cols = self.availCols()
            choice = np.random.choice(avail_cols)
            self.play(color, choice)
        
    def to_tuple(self):
        '''
        Convert board to tuple that can be used to index the state in a lookup table
        '''        
        return tuple(map(tuple, self.board))
        
    def __repr__(self):
        '''
        Pretty print the board
        '''
        acc=["--------- BOARD ----------",str(self.board),"--------------------------"]
        return '\n'.join(acc)

if __name__=="__main__":

    # Generate a random board
    print 'Btw, here is a random board'
    board = Board()
    board.randomize()
    print board
    
    print 'And its flipped version'
    board2 = board.clone()
    board2.flip()
    print board2