import numpy as np
from pprint import pprint as pp

def isDiagonallyDominant(M: np.ndarray):
    """
    For a Matrix M(nxn): |m(i,i)| >= sum(|m(i,j)|, for j in range(n)) for all i in [1,n]. -> Diagonally Dominant By Rows
    For a Matrix M(nxn): |m(j,j)| >= sum(|m(i,j)|, for i in range(n)) for all j in [1,n]. -> Diagonally Dominant By Columns

    :return: (byRows, byCols) 
    """
    n = M.shape[0]
    byRows = True
    byCols = True

    for j in range(n):
        s = 0
        for i in range(n):
            if i == j:
                continue
            s += abs(M[i, j])
        if abs(M[j, j]) <= s:
            byRows = False

    for i in range(n):
        s = 0
        for j in range(n):
            if i == j:
                continue
            s += abs(M[i, j])
        if abs(M[i, i]) <= s:
            byCols = False

    return byRows, byCols


# Matrix Norms
def frobeniusNorm(M: np.ndarray):
    """
    ||A||f = sqrt(sum of squares of all the elements in the matrix) for all j in [1,n].
    """
    return np.sqrt(np.sum(M * M))


def rowSum(M: np.ndarray):
    """
    ||A||∞ = max(sum(rows))
    """
    s = np.zeros(M.shape[0], dtype=M.dtype)
    for i in M:
        s += i
    return max(s)


def colSum(M: np.ndarray):
    """
    ||A||1 = max(sum(columns))
    """
    s = np.zeros((M.shape[0], 1), dtype=M.dtype)
    for i in range(M.shape[0]):
        s += M[:, i]
    return np.max(s)


# Iterative Methods
def conditionNumber(M: np.ndarray):
    return np.linalg.cond(M)


def jacobi(M: np.ndarray, b: np.ndarray, iterLimit: int = 10000, mode: str = 'matrix', printIters: bool = False):
    """
    mode: 'matrix' | 'component'
    """
    iterCount = 0
    m, n = M.shape
    L = np.tril(M, -1)  # Lower Triangle Matrix
    U = np.triu(M, 1)  # Upper Triangle Matrix
    D = np.diag(np.diag(M))

    x = np.zeros((m, 1))
    if mode == 'matrix':
        for i in range(iterLimit):
            if printIters:
                print(f"Iteration {iterCount}")
                print(x)
                print('-' * 30)
                iterCount += 1

            xk = np.linalg.inv(D).dot(b - L.dot(x) - U.dot(x))
            x = xk.copy()
    elif mode == 'component':
        for itr in range(iterLimit):
            if printIters:
                print(f"Iteration {iterCount}")
                pp(x)
                print('-' * 30)
                iterCount += 1

            xk = x.copy()
            for i in range(m):
                lower = 0
                for j in range(i):
                    lower += L[i, j] * x[j, 0]
                upper = 0
                for j in range(i + 1, n):
                    upper += U[i, j] * x[j, 0]
                xk[i, 0] = (b[i, 0] - lower - upper) / D[i, i]
            x = xk.copy()

    if printIters:
        print(f"Total Numeber of Iterations: {iterCount}")
    return x


def gauss_seidel(M: np.ndarray, b: np.ndarray, iterLimit: int = 10000, mode: str = 'matrix', printIters: bool = False):
    """
    mode: 'matrix' | 'component'
    """
    iterCount = 0
    m, n = M.shape
    L = np.tril(M, -1)  # Lower Triangle Matrix
    U = np.triu(M, 1)  # Upper Triangle Matrix
    D = np.diag(np.diag(M))

    # x = np.array([[1.6, -0.8, 1.4]]).reshape(m, 1)
    x = np.zeros((m, 1))
    if mode == 'matrix':
        for i in range(iterLimit):
            if printIters:
                print(f"Iteration {iterCount}")
                print(x)
                print('-' * 30)
                iterCount += 1

            xk = np.linalg.inv(D + L) @ (b - U.dot(x))
            x = xk.copy()
    elif mode == 'component':
        for itr in range(iterLimit):
            if printIters:
                print(f"Iteration {iterCount}")
                print(x)
                print('-' * 30)
                iterCount += 1

            xk = x.copy()
            for i in range(m):
                lower = 0
                for j in range(i):
                    lower += L[i, j] * xk[j, 0]
                upper = 0
                for j in range(i + 1, n):
                    upper += U[i, j] * x[j, 0]
                xk[i, 0] = (b[i, 0] - lower - upper) / D[i, i]
            x = xk.copy()
    if printIters:
        print(f"Total Numeber of Iterations: {iterCount}")
    return x


if __name__ == "__main__":
    # A = np.array([[5, 1, 2], [1, 3, 1], [-1, 2, 4]], dtype=np.int32)
    # B = np.array([[13, 12, 8]], dtype=np.int32).reshape((3, 1))
    # print(jacobi(A, B, iterLimit=10, mode='component', printIters=True))
    # print(gauss_seidel(A, B, iterLimit=1000, mode='component'))
    # print(conditionNumber(np.matrix('20,1,-2;3,20,-1;2,-3,20')))
    pass
