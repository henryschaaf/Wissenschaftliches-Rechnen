
import numpy as np
import matplotlib.pyplot as plt
import datetime

import unittest
import tomograph
from main import compute_tomograph, gaussian_elimination, compute_cholesky, solve_cholesky, is_lowerDreickesMatrix, setup_system_tomograph


class Tests(unittest.TestCase):
    def test_gaussian_elimination(self):
        #A = np.random.randn(4, 4)
        A = np.array([[1,3,4,5,6],[3,4,5,5,7],[2,3,4,5,7],[10,5,6,2,9],[6,7,7,9,40]],dtype=float)
        x = np.random.rand(4)
        b = np.dot(A, x)
        A_elim, b_elim = gaussian_elimination(A, b)
        self.assertTrue(np.allclose(np.linalg.solve(A_elim, b_elim), x))  # Check if system is still solvable
        self.assertTrue(np.allclose(A_elim, np.triu(A_elim)))  # Check if matrix is upper triangular

    def test_back_substitution(self):
        A = np.array([11,44,1],[0.1,0.4,3],[11,44,1])
        # TODO

    def test_cholesky_decomposition(self):
        A = np.array([[8,2,5,7],[2,9,16,6],[5,16,47,34],[7,6,34,80]],dtype=float)
        compute_cholesky(A)
        # TODO

    def test_solve_cholesky(self):
        n_shots = 9  # 128
        n_rays = 18  # 128
        n_grid = 9  # 64

        setup_system_tomograph(n_shots, n_rays, n_grid)
        #self.assertTrue(np.array_equal(solve_cholesky(L,b) ,x)) 
        # TODO

    def test_compute_tomograph(self):
        t = datetime.datetime.now()
        print("Start time: " + str(t.hour) + ":" + str(t.minute) + ":" + str(t.second))

        # Compute tomographic image
        n_shots = 64  # 128
        n_rays = 64  # 128
        n_grid = 32  # 64
        tim = compute_tomograph(n_shots, n_rays, n_grid)

        t = datetime.datetime.now()
        print("End time: " + str(t.hour) + ":" + str(t.minute) + ":" + str(t.second))

        # Visualize image
        plt.imshow(tim, cmap='gist_yarg', extent=[-1.0, 1.0, -1.0, 1.0],
                   origin='lower', interpolation='nearest')
        plt.gca().set_xticks([-1, 0, 1])
        plt.gca().set_yticks([-1, 0, 1])
        plt.gca().set_title('%dx%d' % (n_grid, n_grid))

        plt.show()


if __name__ == '__main__':
    unittest.main()

