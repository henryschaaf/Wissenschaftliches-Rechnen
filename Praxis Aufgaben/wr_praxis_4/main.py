import numpy as np


####################################################################################################
# Exercise 1: Interpolation

def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """
    
    assert (x.size == y.size)

    polynomial = np.poly1d(0)
    base_functions = []

    print("\n")
    print("x wert: ")
    print(x)
    print("y wert: ")
    print(y)
    print("\n")

    # TODO: Generate Lagrange base polynomials and interpolation polynomial
    for i in range(len(x)):
        base_functions.append(1)
        for j in range(len(x)):
            if(i == j):
               continue
            base_functions[i] *= np.poly1d((1,-x[j]))/(x[i]-x[j])
            print("\n")
            print("base")
            print(base_functions)
        
        
        polynomial = polynomial + (y[i] * base_functions[i])
        print("\n")
        print("poly")
        print(polynomial)

    return polynomial, base_functions




def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points
    yp: derivative values of interpolation points

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size == yp.size)

    spline = []
    # TODO compute piecewise interpolating cubic polynomials
    for i in range(0,x.size - 1):
        A = np.array([[1,x[i],x[i] ** 2, x[i] ** 3],[1,x[i+1],x[i+1] ** 2, x[i+1] ** 3],[0,1,2 * x[i], 3 * x[i] ** 2],[0,1,2 * x[i + 1], 3 * x[i + 1] ** 2]])
        b = np.array([y[i],y[i+1],yp[i],yp[i+1]])
        print(A)
        print("\n")
        print(b)
        print("\n")
        poly = np.poly1d(np.flip(np.linalg.solve(A,b)))
        spline.append(poly)
    return spline



####################################################################################################
# Exercise 2: Animation

def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Intepolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    # TODO construct linear system with natural boundary conditions
    A = np.zeros(((4* (x.size - 1))-1,(4* (x.size - 1))))
    b = []
    print(len(b))
    print(A)
    for i in range(0,x.size - 1):
        A[i * 4,i * 4:i * 4 + 4] = [1,x[i],x[i] ** 2, x[i] ** 3]
        A[i * 4 + 1,i * 4:i * 4 + 4] = [1,x[i+1],x[i+1] ** 2, x[i+1] ** 3]
        b.append(y[i])
        b.append(y[i+1])

        if(i != x.size - 2):
            A[i * 4 + 2,i * 4:i * 4 + 8] = [0,1,2 * x[i+1], 3 * x[i+1] ** 2,0,-1,-2 * x[i+1],-3 * x [i+1] ** 2]
            A[i * 4 + 3,i * 4:i * 4 + 8] = [0,0,2, 6 * x[i+1],0,0,-2 ,-6 * x [i+1]]
            b.append(0)
            b.append(0)
        np.set_printoptions(suppress=True)
        print(x)
        print(A)
    print("ende")
    
    c = np.zeros((4* (x.size - 1)))
    b.insert(0,0)
    b.append(0)
    c[0:4] = [0,0,2,6 * x[0]]
    A = np.insert(A,0,c,axis=0)
    A[A.shape[0]-1,A.shape[0]-4:A.shape[0]] = [0,0,2,6 * x[x.size - 1]]
        
        

        
    print(A)
        #A = np.array([[1,x[i],x[i] ** 2, x[i] ** 3,0,0,0,0],[1,x[i+1],x[i+1] ** 2, x[i+1] ** 3,0,0,0,0],[0,1,2 * x[i+1], 3 * x[i+1] ** 2,0,-1,-2 * x[i+1],-3 * x [i+1] ** 2],[0,0,2, 6 * x[i+1],0,0,-2 ,-6 * x [i+1]],[0,0,0,0,1,x[i+1],x[i+1] ** 2, x[i+1] ** 3],[0,0,0,0,1,x[i+2],x[i+2] ** 2, x[i+2] ** 3]])
        #b = np.array([y[i],y[i+1],yp[i],yp[i+1]])
    print(A)
    
    # TODO solve linear system for the coefficients of the spline
    print("\n")
    print("ACHTUNG")
    print(A)
    print("\n")
    print(b)

    
    list_of_coeffs = np.linalg.solve(A,b)
    print(list_of_coeffs)
    print(len(list_of_coeffs))

    spline = []
    # TODO extract local interpolation coefficients from solution
    for i in range(0,x.size - 1):
        print(list_of_coeffs[4*i:4*i+4])
        poly = np.poly1d(np.flip(list_of_coeffs[4*i:4*i+4]))
        print(poly)
        spline.append(poly)
        print(spline[i])
    #print(spline)
    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    # TODO: construct linear system with periodic boundary conditions

    A = np.zeros(((4* (x.size - 1)),(4* (x.size - 1))))
    b = []
    print(len(b))
    print(A)
    for i in range(0,x.size - 1):
        A[i * 4,i * 4:i * 4 + 4] = [1,x[i],x[i] ** 2, x[i] ** 3]
        A[i * 4 + 1,i * 4:i * 4 + 4] = [1,x[i+1],x[i+1] ** 2, x[i+1] ** 3]
        b.append(y[i])
        b.append(y[i+1])

        if(i != x.size - 2):
            A[i * 4 + 2,i * 4:i * 4 + 8] = [0,1,2 * x[i+1], 3 * x[i+1] ** 2,0,-1,-2 * x[i+1],-3 * x [i+1] ** 2]
            A[i * 4 + 3,i * 4:i * 4 + 8] = [0,0,2, 6 * x[i+1],0,0,-2 ,-6 * x [i+1]]
            b.append(0)
            b.append(0)
        np.set_printoptions(suppress=True)
        print(x)
        print(A)
    print("ende")
    
    
    
    b.append(0)
    b.append(0)
    
    A[A.shape[0]-2,A.shape[1]-4:A.shape[1]] = [0,-1,-2 * x[x.size - 1],-3 * x[x.size - 1] ** 2]
    A[A.shape[0]-2,0:4] = [0,1,2 * x[0],3 * x[0] ** 2]
    A[A.shape[0]-1,A.shape[1]-4:A.shape[1]] = [0,0,-2,-6 * x[x.size - 1]]
    A[A.shape[0]-1,0:4] = [0,0,2,6 * x[0]]

    print(A)
    # TODO solve linear system for the coefficients of the spline

    list_of_coeffs = np.linalg.solve(A,b)

    spline = []
    # TODO extract local interpolation coefficients from solution

    for i in range(0,x.size - 1):
        print(list_of_coeffs[4*i:4*i+4])
        poly = np.poly1d(np.flip(list_of_coeffs[4*i:4*i+4]))
        print(poly)
        spline.append(poly)
        print(spline[i])

    print(spline)
    return spline


if __name__ == '__main__':

    x = np.array( [1.0, 2.0, 3.0, 4.0])
    y = np.array( [3.0, 2.0, 4.0, 1.0])

    splines = natural_cubic_interpolation( x, y)

    # # x-values to be interpolated
    # keytimes = np.linspace(0, 200, 11)
    # # y-values to be interpolated
    # keyframes = [np.array([0., -0.05, -0.2, -0.2, 0.2, -0.2, 0.25, -0.3, 0.3, 0.1, 0.2]),
    #              np.array([0., 0.0, 0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.2, -0.3, 0.3])] * 5
    # keyframes.append(keyframes[0])
    # splines = []
    # for i in range(11):  # Iterate over all animated parts
    #     x = keytimes
    #     y = np.array([keyframes[k][i] for k in range(11)])
    #     spline = natural_cubic_interpolation(x, y)
    #     if len(spline) == 0:
    #         animate(keytimes, keyframes, linear_animation(keytimes, keyframes))
    #         self.fail("Natural cubic interpolation not implemented.")
    #     splines.append(spline)

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
