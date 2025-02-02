# test_smith_wilson.py
import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.smith_wilson import SmithWilson

class TestSmithWilson(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
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
        
        self.sw = SmithWilson()
        
        # Expected spot rates (in decimal form)
        self.expected_spots = np.array([
            0.009199, 0.010607, 0.011009, 0.011006, 
            0.013885, 0.017124, 0.019206, 0.020517, 
            0.021504, 0.022428
        ])

    def test_spot_rates(self):
        """Test if calculated spot rates match expected values."""
        result, alfa = self.sw.smith_wilson_brute_force(
            data=self.sample_data,
            cra=0,
            ufr_ac=0.042,
            alfa_min=0.05,
            tau=1,
            t2=60
        )
        
        # Extract spot rates from result (assuming they're in column 1)
        calculated_spots = result[1:11, 2]  # Get first 10 years of spot rates
        
        # Compare with expected values
        for i, (calc, exp) in enumerate(zip(calculated_spots, self.expected_spots)):
            self.assertAlmostEqual(
                calc, 
                exp, 
                places=4, 
                msg=f"Spot rate mismatch at year {i+1}: expected {exp:.4%}, got {calc:.4%}"
            )

if __name__ == '__main__':
    unittest.main()

if __name__ == "__main__":
    unittest.main()