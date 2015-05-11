""" A board for Connect Four.
    Currently does not work for diagonal wins (e.g. 5 in 
    a row along a diagonal).

    """

import numpy as np
import random

class Board():
    # The idea is that you place your pieces at the top, and they
    # drop down to populate the first empty square closest to the bottom.

    def __init__(self, rows=6, cols=7):
        self.board = np.zeros((rows, cols))
        self.nrows = rows
        self.ncols = cols

    """ For playing pieces """

    def __playPiece(self, piece_color, col, zero_index):
        # Private method. 0-indexed
        # Place a piece (denoted by 1) at a given column.

        color = 1 if piece_color == "red" else 2

        # Iterate bottom up until hit a 0; otherwise invalid move!
        for sx in xrange(self.nrows-1, -2, -1):
            if sx == -1:
                # if hit top of board without empty square
                raise Exception("No more moves in that column!")
            if self.board[sx, col] == 0:
                self.board[sx, col] = color
                break

        # Return winner information (1 = RED, 2 = BLACK, -1 = neither yet)
        return self.isSolved()

    def playRed(self, col, zero_index=False):
        # Place a RED piece (denoted by 1) at a given column.
        # Return status of game.
        return self.__playPiece("red", col, zero_index)

    def playBlack(self, col, zero_index=False):
        # Place a BLACK piece (denoted by 2) at a given column.
        # Return status of game.
        return self.__playPiece("black", col, zero_index)

    """ For checking whether the board is solved """

    def __hasFourInRow(self, vector):
        # Given a vector, see if it has a winner (4 of same piece in row).
        # Could be improved and made more concise with NumPy

        # Handle 0's. Note that if >2 spaces are still empty (== 0), you
        # can't have a winner.
        num_zeros = sum([1 if i == 0 else 0 for i in vector])
        if num_zeros > 2:
            return -1

        # Simple linear algorithm to check if there is a 5-in-a-row,
        # and if so return 1 (RED) or 2 (BLACK). Else, return -1.
        this_col = vector
        slow = fast = 0
        while True:
            if this_col[slow] == this_col[fast]:
                fast += 1
            else:
                if abs(fast - slow) >= 4:
                    if this_col[slow] == 1:
                        return 1
                    return 2
                slow = fast
                fast += 1
            if fast > len(this_col) - 1:
                if this_col[slow] == this_col[fast - 1]:
                    if abs(fast - slow) >= 4:
                        if this_col[slow] == 1:
                            return 1
                        return 2
                break
        return -1

    def isSolved(self):
        # Return 1 if RED has won, 2 if BLACK has, and -1 if neither
        # has won yet.

        # Check if any of the columns have winners.
        for i in range(self.ncols):
            result = self.__hasFourInRow(self.board[:,i])
            if result > 0:
                return result

        # Check if any of the rows have winners.
        for j in range(self.nrows):
            result = self.__hasFourInRow(self.board[j,:])
            if result > 0:
                return result

        return -1

    def availCols(self):
        # Return whichever columns are available for more moves.
        return [i for i in xrange(self.ncols) if self.board[0,i] == 0]

    """ Other utilities """

    def getCol(self, col):
        return self.board[:, col]

    def getRow(self, row):
        return self.board[row, :]

    def randomize(self):
        # Randomize the board with 0's, 1's, and 2's, still noting
        # the effects of gravity (pieces fall to bottom). Does not
        # check if the board is solved.

        # Reset board, then generate # of pieces to play.
        self.board = np.zeros((self.nrows, self.ncols))
        num_pieces = random.choice(xrange(self.nrows * self.ncols))
        pieces = iter([random.randint(1,2) for i in xrange(num_pieces)])

        # Reset board. For each piece, check which cols are available to play,
        # and then play whoever's turn it is next.
        for px, piece in enumerate(pieces):
            avail_cols = [i for i in xrange(self.ncols)
                if self.board[0,i] == 0]
            rand_col = random.choice(avail_cols)

            if px % 2 == 0: # whoever goes first
                self.playRed(rand_col)
            else:
                self.playBlack(rand_col)

    def show(self):
        # Pretty print the board
        print "--------- BOARD ----------"
        for j in xrange(self.nrows):
            for i in xrange(self.ncols):
                print "%d " % self.board[j,i],
            print
        print "--------------------------"

if __name__=="__main__":

    # Run a sample game. Both sides here just play randomly.
    nrows, ncols = 6, 7
    board = Board(rows=nrows, cols=ncols)
    for i in xrange(64):
        print "Next: move %d. %s\'s turn!" % (i,
            ("RED" if i % 2 == 0 else "BLACK"))

        # Get a random available column
        avail_cols = board.availCols()
        if len(avail_cols) == 0:
            print "NEITHER RED OR BLACK WIN\n"
            break
        rand_col = random.choice(avail_cols)

        if i % 2 == 0:
            game_status = board.playRed(rand_col)
        else:
            game_status = board.playBlack(rand_col)
        board.show()
        print 
        if game_status > 0:
            print "%s wins!\n" % ("RED" if game_status == 1 else "BLACK")
            break

