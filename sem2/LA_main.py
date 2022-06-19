"""This Program Provides Matrix Functionality of a single matrix"""
from LA_Matrices import *
from sympy import Matrix

MATRIX_LABEL = 'A'
M = None
P = None
D = None
AUG_M = None
TYPE = int
NP_TYPE = np.int16
EXIT = False
EIG_VALS = []
b = None


def welcomeText():
    print('Available Options are: ')
    options = [
        'Exit the Utility',
        'Initialize Matrix',
        'Print Matrix',
        'Basic functionality(power,rank,transpose)',
        'Print Determinant of Matrix',
        'Print Characteristic equation of Matrix',
        'Print Inverse of Matrix',
        'Print Row Reduced Echelon form of Matrix',
        'Print Eigen Values of Matrix',
        'Print Eigen Vectors of Matrix',  # print (output matrix each column is eigen vector)
        'Print the Model Matrix of Matrix',
        'Print the Diagonal Matrix of Matrix',
        'Create Augment Matrix',
        'Print Augment Matrix',
        'Print Row Reduced Echelon form of Augmented Matrix',
        'Print Solutions of Augmented Matrix'
    ]
    for number, command in enumerate(options):
        print(f'{number}. {command} {MATRIX_LABEL}')

    print()
    return int(input('Enter Your Choice of Number: '))


if __name__ == '__main__':
    print('Welcome to LurkingRyuu\'s Matrix Utility!')
    while True:
        try:
            choice = welcomeText()
        except ValueError:
            print('Invalid Input')
            continue

        if choice not in (0, 1, 3) and M is None:
            print('Enter a valid rational choice')
            continue

        if choice < 1:
            EXIT = True
        print()
        match choice:
            case 1:
                matrix_type = input(
                    'Please Enter the Type of Matrix (int/float/complex/ press Enter to Default it to int):')
                if matrix_type == 'float':
                    TYPE = float
                    NP_TYPE = np.float64
                if matrix_type == 'complex':
                    TYPE = complex
                    NP_TYPE = np.complex128
                else:
                    TYPE = int
                    NP_TYPE = np.int16
                MATRIX_LABEL = input('Enter the Label of Matrix (press Enter to Default it to A)  :')
                if MATRIX_LABEL == '' or ord(MATRIX_LABEL) > ord('Z') or ord(MATRIX_LABEL) < ord('A'):
                    MATRIX_LABEL = 'A'
                M = setMatrix(type=TYPE)

            case 2:
                printPrettyMatrix(M, f'Matrix {MATRIX_LABEL}')

            case 3:
                print('Choose an operation')
                print('1.Power the Matrix and print result')
                print('2.Print Rank of the Matrix')
                print('3.Print Transpose of the Matrix')
                c = int(input('Enter Your Choice Number: '))
                if c == 1:
                    n = int(input('Enter the power to which matrix is to be raised: '))
                    printPrettyMatrix(M ** n, f'Transpose of Matrix {MATRIX_LABEL}')
                if c == 2:
                    print()
                    print('-'*50)
                    print(f'Rank of the Matrix {MATRIX_LABEL} is: {M.rank()}')
                    print('-'*50)
                if c == 3:
                    printPrettyMatrix(M.transpose(), f'Transpose of Matrix {MATRIX_LABEL}')

            case 4:
                if M.is_square:
                    print(f'The determinant of Matrix {MATRIX_LABEL} = {M.det()}')
                else:
                    print(f"Matrix {MATRIX_LABEL} is not A Square Matrix")

            case 5:
                if M.is_square:
                    print(f'Characteristic Equation of A: {M.charpoly().as_expr()}=0')
                else:
                    print(f"Matrix {MATRIX_LABEL} is not A Square Matrix")

            case 6:
                if M.det() != 0 and M.is_square:
                    printPrettyMatrix(M.inv(), f'Inverse of Matrix {MATRIX_LABEL}')
                else:
                    print(f"Matrix {MATRIX_LABEL} is not Invertible")

            case 7:
                printPrettyMatrix(M.rref()[0], f'Echelon form of Matrix {MATRIX_LABEL}')

            case 8:
                if M.is_square:
                    EIG_VALS = []
                    # print(M.eigenvals()) # Debugging
                    for value, multiply in M.eigenvals().items():
                        for j in range(multiply):
                            EIG_VALS.append(TYPE(value))
                    print(f'The Eigen Values of Matrix {MATRIX_LABEL} are {EIG_VALS}')
                else:
                    print(f"Matrix {MATRIX_LABEL} is not A Square Matrix")

            case 9:
                print(M.eigenvects())
                if M.is_square:
                    print('-'*50)
                    for eigen_value, multiplicity, eigen_vectors in M.eigenvects():
                        print(f'Eigen Value: {eigen_value}')
                        print(f'Eigen Vectors: ')
                        for i in range(multiplicity):
                            printPrettyMatrix(eigen_vectors[i], f'Eigen Vector {i+1} for Eigen Value {eigen_value}')
                    print('-'*50)
                else:
                    print(f"Matrix {MATRIX_LABEL} is not A Square Matrix")

            case 10:
                if M.is_square and M.is_diagonalizable():
                    printPrettyMatrix(M.diagonalize()[0], f'Model Matrix of Matrix {MATRIX_LABEL}')
                elif not M.is_square:
                    print(f"Matrix {MATRIX_LABEL} is not A Square Matrix")
                else:
                    print(f"Matrix {MATRIX_LABEL} is not Diagonalizable")

            case 11:
                if M.is_square and M.is_diagonalizable():
                    printPrettyMatrix(M.diagonalize()[1], f'Diagonal Matrix of Matrix {MATRIX_LABEL}')
                elif not M.is_square:
                    print(f"Matrix {MATRIX_LABEL} is not A Square Matrix")
                else:
                    print(f"Matrix {MATRIX_LABEL} is not Diagonalizable")

            case 12:
                in_str = input(f'Enter the {M.shape[0]} numbers to augment it to Matrix {MATRIX_LABEL}: ')
                b = Matrix(M.shape[0], 1, list(map(int, in_str.split())))
                try:
                    AUG_M = M.row_join(b)
                except:
                    print('Do it Properly')

            case 13:
                printPrettyMatrix(AUG_M, f'Augmented Matrix of Matrix {MATRIX_LABEL}')

            case 14:
                printPrettyMatrix(AUG_M.rref()[0], f'Echelon form of Augmented Matrix {MATRIX_LABEL}')

            case 15:
                if b is None:
                    in_str = input(f'Enter the {M.shape[0]} numbers to augment it to Matrix {MATRIX_LABEL}: ')
                    b = Matrix(M.shape[0], 1, list(map(int, in_str.split())))
                print(np.array(M))
                print(np.array(b))
                solution = np.linalg.pinv(np.array(M, dtype=np.float64)) @ np.array(b)
                printPrettyMatrix(solution, f'Solutions {MATRIX_LABEL}')

            case _:
                EXIT = True
        print()
        print("*" * 50)
        if EXIT:
            print('Thank You For Using My Utility\t\t\t-Karthikeya A.K.A(Lurkingryuu)')
            exit(0)
