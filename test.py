__maintainer__ = ['CS540-testers-SP22']
__author__ = ['Nicholas Beninato(SP21)', 'Jesus Vazquez(SP22)']
__credits__ = ['Harrison Clark(SP21)', 'Stephen Jasina(SP21)', 'Saurabh Kulkarni(SP21)','Alex Moon(SP21)']
__version__ = 'v0.2.2-modifiedSpring2022'

import sys
import unittest
from time import time
import numpy as np
from hw3 import load_and_center_dataset, get_covariance, get_eig, \
		get_eig_prop, project_image, display_image

data_path = 'YaleB_32x32.npy'

def timeit(func):
    def timed_func(*args, **kwargs):
        t0 = time()
        out = func(*args, **kwargs)
        print(f'Ran {func.__name__}{" "*(30-len(func.__name__))}in {(time() - t0)*1000:.2f}ms')
        return out
    return timed_func


class TestLoadAndCenterDataset(unittest.TestCase):
	@timeit
	def test_load(self):
		x = load_and_center_dataset(data_path)

		# The dataset needs to have the correct shape
		self.assertEqual(np.shape(x), (2414, 1024))

		# The dataset should not be constant-valued
		self.assertNotAlmostEqual(np.max(x) - np.min(x), 0)
        
	@timeit
	def test_center(self):
		x = load_and_center_dataset(data_path)

		# Each coordinate of our dataset should average to 0
		for i in range(np.shape(x)[1]):
			self.assertAlmostEqual(np.sum(x[:, i]), 0)

class TestGetCovariance(unittest.TestCase):
	@timeit
	def test_shape(self):
		x = load_and_center_dataset(data_path)
		S = get_covariance(x)

		# S should be square and have side length d
		self.assertEqual(np.shape(S), (1024, 1024))

	@timeit
	def test_values(self):
		x = load_and_center_dataset(data_path)
		S = get_covariance(x)

		# S should be symmetric
		self.assertTrue(np.all(np.isclose(S, S.T)))

		# S should have non-negative values on the diagonal
		self.assertTrue(np.min(np.diagonal(S)) >= 0)

class TestGetEig(unittest.TestCase):
	@timeit
	def test_small(self):
		x = load_and_center_dataset(data_path)
		S = get_covariance(x)
		Lambda, U = get_eig(S, 2)

		self.assertEqual(np.shape(Lambda), (2, 2))
		self.assertTrue(np.all(np.isclose(
				Lambda, [[1369142.41612494, 0], [0, 1341168.50476773]])))

		# The eigenvectors should be the columns
		self.assertEqual(np.shape(U), (1024, 2))
		self.assertTrue(np.all(np.isclose(S @ U, U @ Lambda)))

	@timeit
	def test_large(self):
		x = load_and_center_dataset(data_path)
		S = get_covariance(x)
		Lambda, U = get_eig(S, 1024)

		self.assertEqual(np.shape(Lambda), (1024, 1024))
		# Check that Lambda is diagonal
		self.assertEqual(np.count_nonzero(
				Lambda - np.diag(np.diagonal(Lambda))), 0)
		# Check that Lambda is sorted in decreasing order
		self.assertTrue(np.all(np.equal(np.diagonal(Lambda),
				sorted(np.diagonal(Lambda), reverse=True))))

		# The eigenvectors should be the columns
		self.assertEqual(np.shape(U), (1024, 1024))
		self.assertTrue(np.all(np.isclose(S @ U, U @ Lambda)))

class TestGetEigProp(unittest.TestCase):
	@timeit
	def test_small(self):
		x = load_and_center_dataset(data_path)
		S = get_covariance(x)
		Lambda, U = get_eig_prop(S,0.07)

		self.assertEqual(np.shape(Lambda), (2, 2))
		self.assertTrue(np.all(np.isclose(
				Lambda, [[1369142.41612494, 0], [0, 1341168.50476773]])))

		# The eigenvectors should be the columns
		self.assertEqual(np.shape(U), (1024, 2))
		self.assertTrue(np.all(np.isclose(S @ U, U @ Lambda)))

	@timeit
	def test_large(self):
		x = load_and_center_dataset(data_path)
		S = get_covariance(x)
		# This will select all eigenvalues/eigenvectors
		Lambda, U = get_eig_prop(S, -1)

		self.assertEqual(np.shape(Lambda), (1024, 1024))
		# Check that Lambda is diagonal
		self.assertEqual(np.count_nonzero(
				Lambda - np.diag(np.diagonal(Lambda))), 0)
		# Check that Lambda is sorted in decreasing order
		self.assertTrue(np.all(np.equal(np.diagonal(Lambda),
				sorted(np.diagonal(Lambda), reverse=True))))

		# The eigenvectors should be the columns
		self.assertEqual(np.shape(U), (1024, 1024))
		self.assertTrue(np.all(np.isclose(S @ U, U @ Lambda)))

class TestProjectImage(unittest.TestCase):
	@timeit
	def test_shape_orig(self):
		x = load_and_center_dataset(data_path)
		S = get_covariance(x)
		_, U = get_eig(S, 2)
		# This is the image of the "9" in the spec
		projected = project_image(x[0], U)

		self.assertEqual(np.shape(projected), (1024,))
		self.assertAlmostEqual(np.min(projected), 0.27875793275517147)
		self.assertAlmostEqual(np.max(projected), 93.22417310945808)
     
	@timeit
	def test_shape_with_two_eig_values(self):
		x = load_and_center_dataset(data_path)
		S = get_covariance(x)
		_, U = get_eig(S, 2)
		# This is the image of the "9" in the spec
		projected = project_image(x[3], U)

		self.assertEqual(np.shape(projected), (1024,))
		self.assertAlmostEqual(np.min(projected), -102.98135151709695)
		self.assertAlmostEqual(np.max(projected), -2.9426401819431263)
     
    # Project_image will be tested with more than 2 eigen values
	@timeit
	def test_shape_with_five_eig_values(self):
		x = load_and_center_dataset(data_path)
		S = get_covariance(x)
		_, U = get_eig(S, 5)
		projected = project_image(x[3], U)

		self.assertEqual(np.shape(projected), (1024,))
		self.assertAlmostEqual(np.min(projected), -25.154139468874448)
		self.assertAlmostEqual(np.max(projected), 17.02338010385981)

	@timeit
	def test_shape_with_ten_eig_values(self):
		x = load_and_center_dataset(data_path)
		S = get_covariance(x)
		_, U = get_eig(S, 10)
		projected = project_image(x[3], U)

		self.assertEqual(np.shape(projected), (1024,))
		self.assertAlmostEqual(np.min(projected), -26.8175300968181)
		self.assertAlmostEqual(np.max(projected), 44.9530102615709)

	@timeit
	def test_shape_with_eig_values_prop(self):
		x = load_and_center_dataset(data_path)
		S = get_covariance(x)
		_, U = get_eig_prop(S,0.02)
		projected = project_image(x[3], U)

        # Projected needs to have shape (1024, )
		self.assertEqual(np.shape(projected), (1024,))
        
        # Example values from implemenation(Should be correct)
		self.assertAlmostEqual(np.min(projected), -43.78744107453002)
		self.assertAlmostEqual(np.max(projected), 42.70786536248303)
        
if __name__ == '__main__':
	# Hack to allow different locations of YaleB_32x32.npy (
    # done this way to allow unittest's flags to still 
    # be passed, if desired)
	if '--data-path' in sys.argv:
		path_index = sys.argv.index('--data-path') + 1
		if path_index == len(sys.argv):
			print('Error: must supply path after option --data-path')
			sys.exit(1)
		data_path = sys.argv[path_index]
		del(sys.argv[path_index])
		del(sys.argv[path_index - 1])

	print('Homework 3 Tester Version', __version__)

	unittest.main(argv=sys.argv)
