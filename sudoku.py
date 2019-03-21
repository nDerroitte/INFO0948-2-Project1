import numpy as np
from copy import deepcopy
import math
from entropy import *

# Constants
SQUARRE = [[0, 2, 0, 8, 0, 0, 0, 3, 0],
           [5, 0, 1, 2, 0, 3, 0, 6, 0],
           [0, 9, 0, 0, 0, 6, 0, 7, 0],
           [0, 0, 1, 5, 4, 0, 0, 0, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [6, 0, 0, 0, 1, 9, 7, 0, 0],
           [0, 9, 0, 2, 0, 0, 0, 1, 0],
           [0, 3, 0, 8, 0, 4, 9, 0, 7],
           [0, 8, 0, 0, 0, 7, 0, 6, 0]]

def getSquarreNumber(grid, index):
    """
    Compute the index of the subgrid of the suduko grid from the index of the
    cell
    """
    i, j = index
    if i < 3 :
        if j < 3:
            return 0
        if 3 <= j < 6:
            return 1
        if 6 <= j < 9:
            return 2
    if 3 <= i < 6:
        if j < 3:
            return 3
        if 3 <= j < 6:
            return 4
        if 6 <= j < 9:
            return 5
    if 6 <= i < 9:
        if j < 3:
            return 6
        if 3 <= j < 6:
            return 7
        if 6 <= j < 9:
            return 8


def getListValueInSquarre(grid, index):
    """
    Get a list containing the cell of the subgrid in which the index is.
    """
    square_index = getSquarreNumber(grid, index)
    return SQUARRE[square_index]


def computeProbabiltyDistribution(grid, index):
    """
    Compute the probability distribution of the cell of the sudoku grid at the
    index given in the parametets.
    """
    # impossible_number will contain all the forbidden digits
    impossible_number = []
    p = np.zeros(9)
    i,j = index
    # First case : the cell is not unkkown
    if grid[i][j]!=0:
        p[grid[i][j]-1] = 1
    # Second case : the cell is known
    else:
        row = grid[i]
        column =  np.transpose(grid)[j]
        square = getListValueInSquarre(grid, index)
        # Verify all the forbidden digits
        for k in range(9):
            if row[k] !=0 and row[k] not in impossible_number:
                impossible_number.append(row[k])
            if column[k] !=0 and column[k] not in impossible_number:
                impossible_number.append(column[k])
            if square[k] !=0 and square[k] not in impossible_number:
                impossible_number.append(square[k])
        # The not forbidden digits are filled by the uniform probability
        proba = 1/ (9-len(impossible_number))
        for k in range(9):
            if k+1 not in impossible_number:
                p[k] = proba
    return p


def computeSudokuEntropy(grid):
    """
    Get the total entropy and the list of all the entropy of the sudoku grid
    """
    # Initialization
    sudoku_entropy = 0
    entropy_list = []
    for i in range(9):
        for j in range(9):
            # Get the probability distribution of each cell
            p = computeProbabiltyDistribution(grid, [i, j])
            # Computing the entropy
            entropy_tmp =  entropy(p)
            # Adding it to the total entropy
            sudoku_entropy += entropy_tmp
            # Adding it to the entropy list
            entropy_list.append(entropy_tmp)
    return entropy_list, sudoku_entropy

if __name__ == "__main__":
    # Loading the sudoku grid
    grid = np.load("sudoku.npy")
    # Getting the entropy list and total entropy
    entropy_list, sudoku_entropy = computeSudokuEntropy(grid)
    print("The entropy of the sudoku is {}.".format(sudoku_entropy))
