import numpy as np
from sympy import symbols, diff
from Function_module import max_value, min_value


def trapezoidal_rule(f, param, lower, upper, n, print_vals = False):
    """
    :param f: The function to be integrated.
    :param a: The lower bound of the integration.
    :param b: The upper bound of the integration.
    :param n: The number of subintervals.
    :return: The integral of f from a to b.
    """
    h = (upper - lower) / n
    if print_vals:
        points = np.linspace(lower, upper, n+1)
        for point in points:
            print(f"f({point}) = {f.evalf(subs={param: point})}")

    return h / 2 * (f.evalf(subs={param: lower}) + f.evalf(subs={param: upper})) + h * sum([f.evalf(subs={param: lower + i * h}) for i in range(1, n)])

def trapezoidal_error(f, param,lower, upper, n, max_error = True, print_error = False):
    """
    :param f: The function to be integrated.
    :param lower: The lower bound of the integration.
    :param upper: The upper bound of the integration.
    :param n: The number of subintervals.
    """
    h = (upper - lower) / n
    g = diff(f, param, 2) * h**3 * n / 12
    if print_error:
        print(f"error = {g}")

    if max_error:
        return max_value(g, param, (lower, upper))
    else:
        return min_value(g, param, (lower, upper))


def simpson_rule(f, param, lower, upper, n, print_vals = False):
    """
    :param f: The function to be integrated.
    :param a: The lower bound of the integration.
    :param b: The upper bound of the integration.
    :param n: The number of subintervals.
    :return: The integral of f from a to b.
    """
    h = (upper - lower) / n
    if print_vals:
        points = np.linspace(lower, upper, n+1)
        for point in points:
            print(f"f({point}) = {f.evalf(subs={param: point})}")
    
    return h / 3 * (f.evalf(subs={param: lower}) + f.evalf(subs={param: upper}) + 4 * sum([f.evalf(subs={param: lower + i * h}) for i in range(1, n, 2)])) + 2 * h / 3 * sum([f.evalf(subs={param: lower + i * h}) for i in range(2, n, 2)])


def simpson_error(f, param, lower, upper, n, max_error = True, print_error = False):
    """
    :param f: The function to be integrated.
    :param lower: The lower bound of the integration.
    :param upper: The upper bound of the integration.
    :param n: The number of subintervals.
    """
    h = (upper - lower) / n
    g = diff(f, param, 4) * h**5 * n / 180
    if print_error:
        print(f"error = {g}")

    if max_error:
        return max_value(g, param, (lower, upper))
    else:
        return min_value(g, param, (lower, upper))


if __name__ == "__main__":
    # Test the trapezoidal rule
    x = symbols('x')

    # f = 0.2 + 25*x - 200*x**2 + 675*x**3 - 900*x**4 + 400*x**5
    
    # prev = 0
    # for i in range(1, 1000):
    #     curr = round(trapezoidal_rule(f, x, 0, 0.8, i), 6)
    #     if abs(curr - prev) < 0.000001:
    #         print(f"Trapezoidal rule converged after {i} intervals.")
    #         print(f"The integral is {curr}.")
    #         break
    #     prev = curr

    f = 1/(3 + 2*x)

    print(f"n=2: {simpson_rule(f, x, 0, 1, 2)}")
    print(f"n=4: {simpson_rule(f, x, 0, 1, 4)}")
