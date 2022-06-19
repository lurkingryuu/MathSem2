from typing import Callable

import numpy as np
import math
from numpy.polynomial.polynomial import Polynomial
from pprint import pprint as pp
from Function_module import Function_NA as F, poly_max
from Function_module import symbols, diff, exp, max_value, min_value, poly_max, poly_min, expand

# <---------- Finding roots using iterative methods ---------->
bisection_count = 0
def bisection_recursion(P, interval: tuple, tolerance: float = 0.0001, iterLim: int = None):
    """The Recursive variant of Bisection method.

    :param P: Polynomial or function of which root is to be found
    :param interval: The interval in which the root is present
    :param tolerance: The error value that can be tolerated, which terminates the iteration
    :param iterLim: It is the iteration limit, if we require the function to return after a certain number of iterations

    :return: root or None
    """
    a, b = interval
    if P(a) * P(b) > 0:
        return None
    if P(a) == 0:
        return a
    if P(b) == 0:
        return b
    mid = (a + b) / 2
    if P(a) * P(mid) < 0:
        b = mid
    else:
        a = mid
    global bisection_count
    if iterLim is not None:
        if abs(P(mid)) < tolerance or iterLim == 0:
            return mid
        iterLim -= 1
        bisection_count += 1
        return bisection(P, (a, b), tolerance, iterLim)
    else:
        if abs(P(mid)) < tolerance:
            return mid
        
        bisection_count += 1
        iterLim -= 1
        return bisection(P, (a, b), tolerance, iterLim)

def bisection(P, interval: tuple, tolerance: float = 0.0001, iterLim: int = None, printIters: bool = False):
    """The Bisection method is one of the iterative methods of finding roots of a polynomial.

    :param P: The polynomial f(x), where the root to be found is of the equation f(x)=0.
    :param interval: The interval in which the root is present
    :param tolerance: The error value that can be tolerated, which terminates the iteration
    :param iterLim: It is the iteration limit, if we require the function to return after a certain number of iterations
    :param printIters: This is a boolean for printing the previous iteration values for the sake of knowing.

    :return root
    """
    a, b = interval
    if P(a) * P(b) > 0:
        return None
    if P(a) == 0:
        return a
    if P(b) == 0:
        return b
    
    for i in range(iterLim):
        mid = (a + b) / 2
        if P(a) * P(mid) < 0:
            b = mid
        else:
            a = mid
        if printIters:
            print(f"iter {i}: {mid}")
        if abs(P(mid)) < tolerance:
            return mid
    return mid
    

# One type of the fixed point iteration
def fixed_point(P, G, inital: float, tolerance: float = 0, iterLim: int = None,
                printIters: bool = False):
    """This iterative method is very popular and efficient but is invalid in various cases.
    x(k+1) = x(k) - f(x(k))/f'(x(k))

    :param P: The polynomial f(x), where the root to be found is of the equation f(x)=0.
    :param G: The function where x = G(x)
    :param inital: The initial x(0) to start the iterations.
    :param tolerance: The error value that can be tolerated, which terminates the iteration.
    :param iterLim: It is the iteration limit, if we require the function to return after a certain number of iterations
    :param printIters: This is a boolean for printing the previous iteration values for the sake of knowing.

    :return root
    :ret: float
    """
    x = inital
    iterCount = 0
    if iterLim is not None:
        while abs(P(x)) > tolerance:
            xk = G(x)
            if iterCount > iterLim or xk == x:
                print(f"x{iterCount} = {xk}")
                break
            if printIters:
                print(f"x{iterCount} = {x}")
            x = xk
            iterCount += 1
    else:
        while abs(P(x)) > tolerance:
            xk = G(x)
            if xk == x:
                print(f"x{iterCount} = {xk}")
                break
            if printIters:
                print(f"x{iterCount} = {x}")
            x = xk
            iterCount += 1
    print(f"Total number of iterations: {iterCount}")
    return x


def newtonRapson(P, P_, inital: float, tolerance: float = 0, iterLim: int = None, printIters: bool = False):
    """This iterative method is very popular and efficient but is invalid in various cases.
    x(k+1) = x(k) - f(x(k))/f'(x(k))

    :param P: The polynomial f(x), where the root to be found is of the equation f(x)=0.
    :param P_: The derivative of the polynomial f(x), also denoted as f'(x).
    :param inital: The initial x(0) to start the iterations.
    :param tolerance: The error value that can be tolerated, which terminates the iteration.
    :param iterLim: It is the iteration limit, if we require the function to return after a certain number of iterations
    :param printIters: This is a boolean for printing the previous iteration values for the sake of knowing.

    :return root
    """
    G: Callable[[float], float] = lambda a: a - P(a) / P_(a)
    return fixed_point(P, G, inital=inital, tolerance=tolerance, iterLim=iterLim, printIters=printIters)


def secant(P, inital: float, second: float, tolerance: float = 0, iterLim: int = None, printIters: bool = False):
    """This iterative method is very popular and efficient but is invalid in various cases.
    x(k+1) = x(k) - f(x(k))/f'(x(k))

    :param P: The polynomial f(x), where the root to be found is of the equation f(x)=0.
    :param inital: The initial x(0) to start the iterations.
    :param second: The x(1) as there should be two initial conditions to start the iterations
    :param tolerance: The error value that can be tolerated, which terminates the iteration.
    :param iterLim: It is the iteration limit, if we require the function to return after a certain number of iterations
    :param printIters: This is a boolean for printing the previous iteration values for the sake of knowing.

    :return root
    """
    G: Callable[[float, float], float] = lambda a, b: a - (P(a)*(a-b)/(P(a)-P(b)))
    x1 = inital
    x2 = second
    xk = x2
    iterCount = 0
    if iterLim is not None:
        while abs(P(xk)) > tolerance:
            xk = G(x2, x1)
            if iterCount > iterLim or xk == x2:
                print(f"x{iterCount} = {xk}")
                break
            if printIters:
                print(f"x{iterCount} = {x1}")
            x1 = x2
            x2 = xk
            iterCount += 1
    else:
        while abs(P(xk)) > tolerance:
            xk = G(x2, x1)
            if xk == x2:
                print(f"x{iterCount} = {xk}")
                break
            if printIters:
                print(f"x{iterCount} = {x1}")
            x1 = x2
            x2 = xk
            iterCount += 1
    print(f"Total number of iterations: {iterCount}")
    return x2


# Polynomial interpolation
def simple_interpolation(points: np.ndarray):
    """This method is used to interpolate a polynomial of degree n-1.
    It is based on the formula: f(x) = a0 + a1*x + a2*x^2 + ... + an-1*x^n-1
    where a0 is the constant term, a1 is the first term, a2 is the second term, ..., an-1 is the n-1th term.

    **** NOTE: This method is highly unstable and should not be used for large n. ****

    :param x: The x values of the points to be interpolated.
    :param y: The y values of the points to be interpolated.
    :return: The polynomial f(x)"""
    n = points.shape[0]
    x = points[:, 0]
    y = points[:, 1]

    b = y.reshape((n, 1))
    A = np.array([x ** i for i in range(n)])
    
    a = np.linalg.solve(A, b)
    
    return lambda x: sum([a[i] * x ** i for i in range(n)])


def newton_interpolation(points: np.ndarray, mode:str = "forward", print_polynomial: bool = False):
    """This method is used to interpolate a polynomial.
    It is a polynomial of the form: a0 + a1*x + a2*x^2 + ... + an*x^n
    
    :param points: The x and y values of the polynomial.
    type points: np.ndarray
    :param mode: The mode of the interpolation. ('forward' | 'backward')
    type: str

    return: The polynomial f(x)
    type: Function_NA
    """
    n = points.shape[0]
    degree = n-1
    X = points[:, 0]
    Y = points[:, 1]

    # Constant difference check
    h = X[1] - X[0]
    for i in range(1, n):
        if X[i] - X[i - 1] != h:
            raise ValueError("The x values are not constant difference.")

    A = np.zeros((n, n))
    A[:, 0] = Y
    for i in range(1, n):
        for j in range(1, i + 1):
            A[i, j] = (A[i, j - 1] - A[i - 1, j - 1])
    
    # return A # giving lower triangle diffs

    f = F()
    f.newton_interp(X, A, mode)

    if print_polynomial:
        print(f"Table: ")
        pp(A)
        f.print_polynomial()

    return f.evalf(subs={f.x: 15})


def newton_error(X: np.ndarray, f, domain: tuple):
    """This method is used to calculate the error of a newton interpolated polynomial.
    
    :param X: The x values of the polynomial.
    type X: np.ndarray
    :param f: The polynomial f(x)
    type f: Sympy polynomial
    """
    n = X.shape[0]
    x = symbols('x')

    g = diff(f(x), x, n)

    poly = math.prod([(x - x_) for x_ in X])
    poly /= math.factorial(n)
    
    err1 = max_value(g, x,domain=domain)
    # err2 = max_value(poly, x,domain=domain) # Very near value to the value given by poly_max
    err2 = poly_max(poly, x,domain=domain)

    return err1 * err2


# Legrange helper for the interpolation
def legrange(i, x, X):
    """This method is used to interpolate a polynomial.
    It is a polynomial of the form: a0 + a1*x + a2*x^2 + ... + an*x^n
    """
    n = X.shape[0]
    x = symbols('x')
    
    poly = math.prod([(x - X[j]) for j in range(n) if j != i])
    poly /= math.prod([(X[i] - X[j]) for j in range(n) if j != i])

    return poly


def legrange_interpolation(points: np.ndarray, print_polynomial: bool = False):
    """This method is used to interpolate a polynomial.
    It is a polynomial of the form: a0 + a1*x + a2*x^2 + ... + an*x^n
    
    :param points: The x and y values of the polynomial.
    type points: np.ndarray
    :param mode: The mode of the interpolation. ('forward' | 'backward')
    type: str

    return: The polynomial f(x)
    type: Function_NA
    """
    n = points.shape[0]
    degree = n-1
    X = points[:, 0]
    Y = points[:, 1]

    f = sum([Y[i] * legrange(i, X, X) for i in range(n)])
    if print_polynomial:
        print(f"Initial: {f}")
    f = expand(f)
    if print_polynomial:
        print(f"Expanded: {f}")

    return f






if __name__ == '__main__':
    # coef = np.array([1, -5, 0, 1])
    # f = Polynomial(coef)

    # Can be reduced to:
    # f_cos = lambda x : np.cos(x) - x * np.exp(x)
    # def f_cox(x: float):
    #     return np.cos(x) - x * np.exp(x)


    # def f_cox_(x: float):
        # return -np.sin(x) - x * np.exp(x) - np.exp(x)


    # root = newtonRapson(P=f, P_=f.deriv(), inital=0.5, iterLim=4, printIters=True)
    # root = newtonRapson(P=f_cox, P_=f_cox_, inital=1, tolerance=1e-8, printIters=True, iterLim=100)
    # print(root)

    # points = np.array([[4, 1], [6, 3], [8, 8], [10, 16]])
    # f = newton_interpolation(points, mode="forward", print_polynomial=True)
    # print(f"f(5)={f(x=5)}")

    # points = np.array([[1, 1], [2, -1], [3, 1], [4, -1], [5, 1]])
    # f = newton_interpolation(points, mode="backward", print_polynomial=True)
    # print(f"f(5)={f(5)}")

    # X = np.array([0, 0.5, 1])
    # e = newton_error(X, exp, (0, 1))
    # print(e)
    
    points = np.array([[0, 1],[1, 14], [2, 15], [4, 5], [5, 6], [6, 19]])
    f = legrange_interpolation(points, print_polynomial=True)
    

