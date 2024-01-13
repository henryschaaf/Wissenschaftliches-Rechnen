
import numpy as np
import tomograph


####################################################################################################
# Exercise 1: Gaussian elimination

def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting.

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """
    # Create copies of input matrix and vector to leave them unmodified
    A = A.copy()
    b = b.copy()

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not

    n_a, m_a = A.shape
    n_b, = b.shape

 

    if n_a != n_b or n_a != m_a:
        raise ValueError("Matrizen passen nicht")
    
    # TODO: Perform gaussian elimination
 

    for i in range(0,m_a):
        
        max = A[i][i]
        init = i
        for j in range(i+1,n_a):
            if A[j][i] > max:
                max = A[j][i]
                init = j

        if use_pivoting == True:
            

            if A[init][i] != 0:
                temp = np.copy(A[init, :])
                temp_b = b[init]
                A[init, :] = A[i, :]
                b[init] = b[i]

                A[i, :] = temp
                b[i] = temp_b
        
        if A[i][i] == 0 and use_pivoting == False:
            raise ValueError("Pivoting empfohlen")
        
        if np.all(A[i,:] == 0):
            raise ValueError("Nullzeile")
        
                    
        
        
        for j in range(i + 1,n_a):
            mult = A[j][i] / A[i][i]
            A[j][:] -= mult * A[i][:]
            b[j] -= mult * b[i]


  
    

    return A, b


def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not

    n_a, m_a = A.shape
    n_b, = b.shape

    if n_a != n_b or n_a != m_a:
        raise ValueError("Matrizen passen nicht")
    
    # TODO: Initialize solution vector with proper size
    x = np.zeros(m_a)

    # TODO: Run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist

    for i in range(m_a - 1, -1, -1):
        if A[i][i] == 0 or b[i] == 0:
            raise ValueError("Keine oder unendlich Lösungen")
        x[i] = (b[i] - np.dot(x[i + 1:],A[i][i + 1:]))/A[i][i]

    return x

def forward_substitution(A: np.ndarray, b: np.ndarray ) -> np.ndarray:
    n_a, m_a = A.shape
    n_b, = b.shape

    if n_a != n_b or n_a != m_a:
        raise ValueError("Matrizen passen nicht")
    
    x = np.zeros(m_a)


    for i in range(0,m_a):
        if A[i][i] == 0 or b[i] == 0:
            raise ValueError("Keine oder unendlich Lösungen")
        x[i] = (b[i] - np.dot(x[:i],A[i][:i]))/A[i][i]

    return x
####################################################################################################
# Exercise 2: Cholesky decomposition

def is_Symmetrisch(M: np.ndarray) -> bool:
    (n, m) = M.shape

    if n != m:
        return False

    for i in range(0,n):
        if not np.all(np.isclose(M[i, i + 1:], M[i + 1:,i], np.finfo(float).eps)):
            return False
        
    return True

def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L : Cholesky multtor of M

    Forbidden:
    - numpy.linalg.*
    """

    # TODO check for symmetry and raise an exception of type ValueError
    (n, m) = M.shape
    
    if not is_Symmetrisch(M):
        raise ValueError("Matrix nicht Symmetrisch")

    

    # TODO build the multtorization and raise a ValueError in case of a non-positive definite input matrix

    L = np.zeros((n, n))

    for j in range(0,n):
        for i in range(j + 1):
            if i == j:
                
                if M[i,i] < 0 or (M[i,i] - (np.sum(L[i, :i] ** 2))) < 0:
                    raise ValueError("Matrix ist keine PD-Matrix")
                
                L[i,i] = np.sqrt(M[i,i] - (np.sum(L[i, :i] ** 2)))
            else:
                L[j,i] = (M[j,i] - np.sum((L[j,:i] * L[i,:i]))) / L[i,i]
            

    return L

def is_lowerDreickesMatrix(M: np.ndarray) -> bool:
   
    n, m = M.shape

    for i in range(0,n):
        if not np.array_equal(M[i,i+1:], np.zeros(n - i - 1)):
            return False
    return True
                          

def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky multtor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """

    # TODO Check the input for validity, raising a ValueError if this is not the case
    (n, m) = L.shape
    n_b, = b.shape

    if n != n_b or n != m:
        raise ValueError("Eingaben passen nicht")
    

    if not is_lowerDreickesMatrix(L):
        raise ValueError("Keine unere Dreiecksmatrix")
    
    
    


    # TODO Solve the system by forward- and backsubstitution
    x = np.zeros(m)

    x = np.transpose(forward_substitution(L,b))

    x = back_substitution(np.transpose(L),x)

    return x


####################################################################################################
# Exercise 3: Tomography

def setup_system_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots : number of different shot directions
    n_rays  : number of parallel rays per direction
    n_grid  : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    L : system matrix
    g : measured intensities

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    -
    """

    # TODO: Initialize system matrix with proper size
    L = np.zeros((n_rays * n_shots, n_grid * n_grid))
    # TODO: Initialize intensity vector
    g = np.zeros(n_rays * n_shots)

    # TODO: Iterate over equispaced angles, take measurements, and update system matrix and sinogram
    theta = 0

    for i in range(0,n_shots):
        intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)
        theta += np.pi / n_shots

        for j in range(0, n_rays):
            g[i * n_rays + j] = intensities[j]

        for j in range(0, len(ray_indices)):
            #g[i * n_rays + ray_indices[j]] = lengths[ray_indices[j]] # unnötig oft aufgerufen

            
            L[i * n_rays + ray_indices[j],isect_indices[j]] = lengths[j]

    
    # Take a measurement with the tomograph from direction r_theta.
    # intensities: measured intensities for all <n_rays> rays of the measurement. intensities[n] contains the intensity for the n-th ray
    # ray_indices: indices of rays that intersect a cell
    # isect_indices: indices of intersected cells
    # lengths: lengths of segments in intersected cells
    # The tuple (ray_indices[n], isect_indices[n], lengths[n]) stores which ray has intersected which cell with which length. n runs from 0 to the amount of ray/cell intersections (-1) of this measurement.
    # intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)

    return [L, g]


def compute_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots : number of different shot directions
    n_rays  : number of parallel rays per direction
    n_grid  : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    tim : tomographic image

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    """

    # Setup the system describing the image reconstruction
    [L, g] = setup_system_tomograph(n_shots, n_rays, n_grid)

    L_transf = np.transpose(L)

    # TODO: Solve for tomographic image using your Cholesky solver
    # (alternatively use Numpy's Cholesky implementation)


    x = solve_cholesky(compute_cholesky(np.dot(L_transf,L)),np.dot(L_transf,g))

    # TODO: Convert solution of linear system to 2D image
    tim = np.zeros((n_grid, n_grid))

    for i in range(0,n_grid):
        for j in range(0,n_grid):
            tim[i,j] = x[i * n_grid + j]

    return tim


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
