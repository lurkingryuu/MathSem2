# Linear Algebra
## Installation
Clone this repository to your project directory.

Run `pip install -r requirements.txt` to install dependencies.

## Documentation
example:
```python
import sem2.NA_Matrix as nam

print(help(nam)) # Module Help
print(help(nam.jacobi)) # function help
```

## Linear Analysis
Download the [ZIP](https://drive.google.com/file/d/1-HTrZrj-Ts6y8ov6_-LnwZn3M3kAQ2cb/view?usp=sharing) file and extract it in your local computer.
Run the LA_main.exe file in it.
Else Run the LA_main.py file in the Module

## Numerical Analysis
### Matrices
import:
```python
import sem2.NA_Matrix as nam
```

The NA_Matrix submodule in the module contains Jacobi and Gauss seidel iteration methods majorly.
Everything here is a numpy array, so be sure to input the proper type if you do not want errors.
You can check the documentation by using `help(NA_Matrix)`.

### Poynomial
import:
```python
import sem2.NA_Polynomial as nap
```

bisection, newtonRapson, fixed_point, secant -> all of them use the inbuilt functions of the python.
example: 
```python
def f(x):
    return x**2 -1 
```
or you could use lambda's:
```python
f = lambda x: x**2 -1 
```

The Interpolation functions, simple_interpolation, newton_interpolation, newton_error, legrange_interpolation use sympy functions.

example of making sympy functions:
```python
import sympy

x = sympy.symbols('x')

f = x**2 - 1
# You can pass this function in the required places.
# There are also other inbuilt functions in sympy.
g = sympy.sin(x)
h = sympy.exp(x)
```
### Integration
import
```python
import sem2.NA_Integration as nai
```

trapezoidal_rule, trapezoidal_error, simpson_rule, simpson_error
are the functions present here.
The same sympy functions are used here.

# Author Details
Yelisetty Karthikeya S M (A.K.A. Lurkingryuu)
</br>
contact: [Gmail](mailto:yelisettikarthik0@gmail.com)
