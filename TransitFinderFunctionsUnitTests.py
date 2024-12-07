import unittest
import numpy as np
from transitFinderFunctions import estimate_planet_radius

class TestTransitFinderFunctions(unittest.TestCase):

    def test_estimate_planet_radius(self):
        # Test case 1: transit_depth = 0.01, stellar_radius = 1 (solar radii)
        self.assertAlmostEqual(estimate_planet_radius(0.01, 1), 0.1)

        # Test case 2: transit_depth = 0.04, stellar_radius = 2 (solar radii)
        self.assertAlmostEqual(estimate_planet_radius(0.04, 2), 0.4)

        # Test case 3: transit_depth = 0.09, stellar_radius = 3 (solar radii)
        self.assertAlmostEqual(estimate_planet_radius(0.09, 3), 0.9)

        # Test case 4: transit_depth = 0.25, stellar_radius = 4 (solar radii)
        self.assertAlmostEqual(estimate_planet_radius(0.25, 4), 2.0)

        # Test case 5: transit_depth = 0.0, stellar_radius = 5 (solar radii)
        self.assertAlmostEqual(estimate_planet_radius(0.0, 5), 0.0)

        # Test case 6: transit_depth = 1.0, stellar_radius = 6 (solar radii)
        self.assertAlmostEqual(estimate_planet_radius(1.0, 6), 6.0)

if __name__ == '__main__':
    unittest.main()