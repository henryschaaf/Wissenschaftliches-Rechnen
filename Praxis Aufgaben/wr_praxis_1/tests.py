
import numpy as np
import unittest
from main import rotation_matrix, matrix_multiplication, compare_multiplication, inverse_rotation, close, machine_epsilon

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
        info_2 = machine_epsilon(np.float32)
        info_float32 = np.finfo(np.float32)
        print(info_2)
        print(info_float32.eps)
        # TODO
        
    def test_is_close(self):
        # TODO
        
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[1.0, 2.0], [3.0, 7.1]])
        self.assertTrue(close(a, b, 3.1))
        
        
    def test_rotation_matrix(self):
        pass
        # TODO

    def test_inverse_rotation(self):
        theta = 90
        expected_inverse_matrix = np.array([[0, 1], [-1, 0]])        
        result_matrix = inverse_rotation(theta)
        self.assertTrue(np.allclose(result_matrix, expected_inverse_matrix))
        # TODO


if __name__ == '__main__':
    unittest.main()
