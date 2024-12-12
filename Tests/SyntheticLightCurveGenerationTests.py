import unittest
import numpy as np
from SyntheticLightCurveGeneration import (
    calculate_keplerian_orbit,
    calculate_limb_darkened_light_curve,
    limb_darken_values,
    generate_random_planet_systems,
    process_system
)

class TestPlanetGenerationExtreme(unittest.TestCase):
    def test_calculate_keplerian_orbit(self):
        period = 2 * np.pi
        transit_midpoint = 0
        semi_major_axis = 1
        time_array = np.linspace(0, 2 * np.pi, 5)
        expected_x = semi_major_axis * np.cos(2 * np.pi * (time_array + transit_midpoint) / period)
        expected_y = semi_major_axis * np.sin(2 * np.pi * (time_array + transit_midpoint) / period)
        x, y = calculate_keplerian_orbit(period, transit_midpoint, semi_major_axis, time_array)
        np.testing.assert_almost_equal(x, expected_x)
        np.testing.assert_almost_equal(y, expected_y)

    def test_calculate_limb_darkened_light_curve(self):
        light_curve = np.ones(5)
        x_value_of_orbit = np.zeros(5)
        y_value_of_orbit = np.linspace(-0.5, 0.5, 5)
        planet_radius = 0.01
        limb_darkening_u1 = 0.8
        limb_darkening_u2 = 0.25
        star_radius = 1.0
        result = calculate_limb_darkened_light_curve(
            light_curve.copy(),
            x_value_of_orbit,
            y_value_of_orbit,
            planet_radius,
            limb_darkening_u1,
            limb_darkening_u2,
            star_radius
        )
        self.assertEqual(len(result), len(light_curve))
        self.assertTrue(np.all(result <= 1))
        self.assertTrue(np.all(result >= 0))

    def test_limb_darken_values(self):
        u1, u2 = limb_darken_values()
        self.assertGreaterEqual(u1, 0.7)
        self.assertLessEqual(u1, 0.9)
        self.assertGreaterEqual(u2, 0.2)
        self.assertLessEqual(u2, 0.3)

    def test_generate_random_planet_systems(self):
        num_systems = 2
        max_planets_per_system = 4
        total_time = 365
        systems = generate_random_planet_systems(num_systems, max_planets_per_system, total_time)
        self.assertEqual(len(systems), num_systems)
        for system in systems:
            self.assertIn('planets', system)
            self.assertTrue(1 <= len(system['planets']) <= max_planets_per_system)
            self.assertIn('star_radius', system)
            self.assertIn('observation_noise', system)
            self.assertIn('u1', system)
            self.assertIn('u2', system)

    def test_process_system(self):
        systems = generate_random_planet_systems(1, 3, 365)
        system = systems[0]
        result = process_system(
            system,
            snr_threshold=5,
            total_time=365,
            cadence=0.0208333
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 11)
        time_array, flux_with_noise, combined_light_curve, total_time_result, star_radius, observation_noise, u1, u2, planets, num_detectable_planets, total_planets = result
        self.assertEqual(len(time_array), len(flux_with_noise))
        self.assertEqual(len(flux_with_noise), len(combined_light_curve))
        self.assertEqual(total_planets, len(system['planets']))
        self.assertIsInstance(num_detectable_planets, int)

if __name__ == '__main__':
    unittest.main()