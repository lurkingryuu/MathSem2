from sympy import *
from sympy.solvers import solve
import numpy as np


class Function_NA:
    def __init__(self):
        self.f = None

    def __call__(self, **kwargs):
        """
        :param kwargs: The keyword arguments to be passed to the function as x=value, y=value, z=value, etc.
        """
        return self.f.evalf(subs=kwargs)
    

    def evalf(self, n: int = 6, **subs):
        return self.f.evalf(n=n, subs=subs)


    def get_function(self):
        return self.f

    def set_function(self, f, **vars):
        self.f = f
        self.vars = vars
    
    def newton_interp(self, X: np.ndarray, A: np.ndarray, mode:str = "forward", round_to: int = 4):
        """
        :param X: The x values of the polynomial.
        :param a: The coefficients.
        :param mode: The mode of the interpolation. ('forward' | 'backward')
        """
        n = X.shape[0]
        h = X[1] - X[0]
        a = np.zeros((n,))

        x = symbols('x')
        self.x = x
        if mode == "forward":
            for i in range(n):
                a[i] = A[i, i]/(factorial(i)*(h**i))
            
            
            terms = np.array([1] + [(x-x_) for x_ in X if x_ != X[-1]])

        elif mode == "backward":
            for i in range(n):
                a[i] = A[n-1, i]/(factorial(i)*(h**i))

            terms = np.array([1] + [x-x_ for x_ in X[::-1] if x_ != X[0]])

        f = 0
        for i in range(n):
            total = round(a[i], round_to)
            for j in range(i+1):
                total *= terms[j]
            f += total

        f = expand(f)
        self.f = f
    
    def print_polynomial(self):
        print(f"f(x) = {self.f}")

# Maximum value of any function
def func_vals(f, param, domain: tuple, accuracy: float = 0.01):
    a, b = domain
    space = np.linspace(a, b, int(abs(b-a)/accuracy))
    return [f.evalf(subs={param: x}) for x in space]

def max_value(f, param, domain: tuple, accuracy: float = 0.01):    
    return max(func_vals(f, param, domain, accuracy))

def min_value(f, param, domain: tuple):
    return min(func_vals(f, param, domain))

def criticals(f, param, domain: tuple):
    a, b = domain
    roots = solve(diff(f, param), param)
    roots += [a, b]
    return [f.evalf(subs={param: x}) for x in roots]

def poly_max(f, param, domain: tuple):
    return max(criticals(f, param, domain))

def poly_min(f, param, domain: tuple):
    return min(criticals(f, param, domain))

