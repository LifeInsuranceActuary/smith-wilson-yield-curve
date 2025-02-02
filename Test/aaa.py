# test_smith_wilson.py
import unittest
import numpy as np
from code.smith_wilson import SmithWilson  # Import the SmithWilson class

class TestSmithWilson(unittest.TestCase):
    def setUp(self):
        """Set up test data and initialize the SmithWilson instance."""
        self.sample_data = np.array([
            [1, 1.0, 0.0092],
            [1, 2.0, 0.0106],
            [1, 3.0, 0.011],
            [1, 4.0, 0.011],
            [1, 5.0, 0.0138], 
            [1, 6.0, 0.0169],
            [1, 10.0, 0.0219],
            [1, 20.0, 0.0295]
        ], dtype=np.float64)
        
        self.sw = SmithWilson()  # Initialize the SmithWilson instance

    def test_smith_wilson_brute_force(self):
        """Test the smith_wilson_brute_force method."""
        result, alfa = self.sw.smith_wilson_brute_force(
            data=self.sample_data,
            cra=0,
            ufr_ac=0.042,
            alfa_min=0.05,
            tau=1,
            t2=60
        )
        
        # Check if the result is a numpy array
        self.assertIsInstance(result, np.ndarray)
        
        # Check if the shape of the result is correct
        self.assertEqual(result.shape, (122, 6))
        
        # Check if alpha is a float and within expected range
        self.assertIsInstance(alfa, float)
        self.assertGreaterEqual(alfa, 0.05)
        
        # Check specific values (e.g., discount factor for year 0 should be 1)
        self.assertAlmostEqual(result[0, 0], 1.0, places=6)
        
        # Check if the discount factors are decreasing over time
        for i in range(1, 30):
            self.assertLess(result[i, 0], result[i - 1, 0])

if __name__ == "__main__":
    unittest.main()