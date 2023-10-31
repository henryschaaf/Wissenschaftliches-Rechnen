
import numpy as np
import unittest
from main import rotation_matrix, matrix_multiplication, compare_multiplication, inverse_rotation

class Tests(unittest.TestCase):

    def test_matrix_multiplication(self):
        a = np.random.randn(2, 2)
        c = np.random.randn(3, 3)
        self.assertTrue(np.allclose(np.dot(a, a), matrix_multiplication(a, a)))
        self.assertRaises(ValueError, matrix_multiplication, a, c)

    def test_compare_multiplication(self):
        r_dict = compare_multiplication(200, 40)
        for r in zip(r_dict["results_numpy"], r_dict["results_mat_mult"]):
            self.assertTrue(np.allclose(r[0], r[1]))

    def test_machine_epsilon(self):
        pass
        # TODO
        
    def test_is_close(self):
        pass
        # TODO
        
    def test_rotation_matrix(self):
        pass
        # TODO

    def test_inverse_rotation(self):
        pass
        # TODO


if __name__ == '__main__':
    unittest.main()
