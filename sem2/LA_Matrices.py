import numpy as np
from numpy import matrix
from sympy import Matrix
from pandas import DataFrame


class Matrices:
    row = 0
    col = 0
    square_matrix = False
    M_sympy = None
    M_numpy = None
    M_model = None
    M_diagonal = None

    def __init__(self, letter: str, mtrx: Matrix = None):
        self.M_sympy = mtrx
        self.letter = letter
        if mtrx is not None:
            self.M_numpy = Matrix(self.M_sympy)
            self.row, self.col = self.M_numpy.shape
            if self.row is self.col:
                self.square_matrix = True
                self.Diagonalise()

    def Diagonalise(self, print=False):
        if self.square_matrix:
            self.M_model, self.M_diagonal = self.M_sympy.diagonalize()
        if print:
            self.printSympyMatrix(self.M_model, 'The model matrix')
            self.printSympyMatrix(self.M_diagonal, 'The Diagonal matrix')

    def getDiagonalMatrix(self):
        return self.M_diagonal

    def getModelMatrix(self):
        return self.M_model

    def setIntegerMatrix(self):
        if not self.M_sympy:
            print("Enter the dimensions as (rows,cols), then start entering the row values in newlines\n")
            # Take in dimensions
            dim = tuple(map(int, input().split()))
            self.row, self.col = dim

            inputList = []
            for i in range(dim[0]):
                while True:
                    l = list(map(complex, input().split()))
                    if len(l) != dim[1]:
                        print(f"Enter the row {i + 1} elements properly")
                        continue
                    rowString = ",".join(list(map(str, l)))
                    break
                inputList.append(rowString)

            inputString = ";".join(inputList)
            self.M_numpy = matrix(inputString.strip())
            self.M_sympy = Matrix(self.M_numpy)
            if self.row is self.col:
                self.square_matrix = True
                self.Diagonalise()

        else:
            print("Matrix Already Assigned")
        return

    def echelon(self):
        # Row redused echelon form
        M_echelon = self.M_sympy.rref()
        # print(M_echelon) --> Prints (Matrix Object, some tuple) 
        return M_echelon[0]

    def printMatrix(self, mtrx: matrix = None, label: str = "No Label"):
        if mtrx is None:
            mtrx = self.M_numpy
        df = DataFrame(mtrx)
        print('-' * (50))
        print(label)
        print('-' * (50))
        print(df.to_string(header=False, index=False))
        print('-' * (50))

    def printSympyMatrix(self, mtrx: Matrix = None, label: str = "No Label"):
        """Prints the matrix in a nice format"""
        if mtrx is None:
            mtrx = self.M_sympy
        df = DataFrame(matrix(mtrx))
        print('-' * (50))
        print(label)
        print('-' * (50))
        print(df.to_string(header=False, index=False))
        print('-' * (50))

    def characteristicPolynomial(self):
        """Prints the characteristic polynomial of the matrix"""
        print(f'Characteristic Equation of A: {self.M_sympy.charpoly().as_expr()}')

    def eigen(self, printMode: str = '', ret: str = ''):
        '''
        printMode: "print values" | "print pairs" 
        ret: 'model matrix' or 'eigen matrix' | 'diagonal matrix'
        '''
        if self.square_matrix:
            eigValues = []
            eigenFull = self.M_sympy.eigenvects()
            for eigVal, algebraicMultiplicity, eigVector in eigenFull:
                eigValues.append(eigVal)

            if printMode == 'print values':
                print(f"Eigen Values of A: {eigValues}")
            if printMode == 'print pairs':
                for eigVal, algebraicMultiplicity, eigVector in eigenFull:
                    self.printMatrix(mtrx=eigVector,
                                     label=f'Eigen Vector of Eigen Value: {eigVal}\nAlgebraic Multiplicity: {algebraicMultiplicity}')

            if ret == 'model matrix' or ret == 'eigen matrix':
                return Matrix(self.M_model)
            if ret == 'diagonal matrix':
                return Matrix(self.M_diagonal)

        else:
            raise Exception('The Matrix is not a square Matrix')


# <----------------- Utility Functions ---------------------->
# Matrix Pretty Print
def printPrettyMatrix(mtrx, label: str = 'No Label'):
    if type(mtrx) is not matrix:
        mtrx = matrix(mtrx)
    df = DataFrame(mtrx)
    print('-' * (50))
    print(label)
    print('-' * (50))
    print(df.to_string(header=False, index=False))
    print('-' * (50))


def setMatrix(type):
    print("Enter the dimensions as (rows,cols), then start entering the row values in newlines")
    # Take in dimensions
    dim = tuple(map(int, input().split()))

    inputList = []
    for i in range(dim[0]):
        while True:
            l = list(map(type, input().split()))
            if len(l) != dim[1]:
                print(f"Enter the row {i + 1} elements properly")
                continue
            rowString = ",".join(list(map(str, l)))
            break
        inputList.append(rowString)

    inputString = ";".join(inputList)
    M_numpy = matrix(inputString.strip())
    M_sympy = Matrix(M_numpy)
    # if row is col:
    #     square_matrix = True
    #     Diagonalise()

    return M_sympy


def printEigenVectors(M):
    eigenFull = M.eigenvects()
    for eigVal, algebraicMultiplicity, eigVector in eigenFull:
        printMatrix(mtrx=eigVector,
                    label=f'Eigen Vector of Eigen Value: {eigVal}\nAlgebraic Multiplicity: {algebraicMultiplicity}')


def printMatrix(mtrx, label: str = "No Label"):
    if type(mtrx) is not np.ndarray:
        mtrx = np.ndarray(mtrx)
    df = DataFrame(mtrx)
    print('-' * (50))
    print(label)
    print('-' * (50))
    print(df.to_string(header=False, index=False))
    print('-' * (50))
