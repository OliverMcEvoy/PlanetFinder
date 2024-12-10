import unittest
import numpy as np
from transitFinderFunctions import (
    run_bls_analysis,
    calculate_fit_for_period,
    estimate_planet_radius,
    analyze_period,
    remove_duplicate_periods
)

class TestTransitFinderFunctions(unittest.TestCase):

    def test_estimate_planet_radius(self):
        transit_depth = 0.01  # 1% transit depth
        stellar_radius = 1.0  # Solar radius
        expected_radius = np.sqrt(transit_depth) * stellar_radius
        calculated_radius = estimate_planet_radius(transit_depth, stellar_radius)
        self.assertAlmostEqual(calculated_radius, expected_radius, places=6)

    def test_analyze_period(self):
        time = np.linspace(0, 30, 1000)
        flux = np.ones_like(time)
        error = np.full_like(time, 0.01)
        period = 10.0
        resolution = 1000
        duration_range = (0.1, 0.5)
        allowed_deviation = 0.05
        result = analyze_period(period, time, flux, error, resolution, duration_range, allowed_deviation)
        self.assertIsNotNone(result)
        self.assertIn("refined_period", result)
        self.assertIn("power", result)

    def test_remove_duplicate_periods(self):
        results_list = [
            {'refined_period': 10.0, 'power': 50},
            {'refined_period': 10.2, 'power': 48},
            {'refined_period': 20.0, 'power': 30},
            {'refined_period': 20.5, 'power': 28},
            {'refined_period': 30.0, 'power': 15},
        ]
        duplicates_percentage_threshold = 0.05
        cleaned_results = remove_duplicate_periods(results_list, duplicates_percentage_threshold)
        expected_periods = {10.0, 20.0, 30.0}
        cleaned_periods = {res['refined_period'] for res in cleaned_results}
        self.assertEqual(cleaned_periods, expected_periods)
    
    def test_run_bls_analysis(self):
        # Mock data
        time = np.linspace(0, 30, 1000)
        # Simulate a transit signal
        flux = np.ones_like(time)
        flux[500:510] -= 0.01  # Inject a transit
        error = np.full_like(time, 0.001)
        resolution = 1000
        min_period = 1.0
        max_period = 15.0
        duration_range = (0.1, 0.5)
        # Run BLS analysis
        results, power, periods, best_period, best_transit_model = run_bls_analysis(
            time, flux, error, resolution, min_period, max_period, duration_range
        )
        self.assertIsNotNone(results)
        self.assertIsNotNone(power)
        self.assertIsNotNone(periods)
        self.assertTrue(len(periods) == resolution)
        self.assertTrue(min_period <= best_period <= max_period)

    def test_calculate_fit_for_period(self):
        result = {
            'refined_period': 10.0,
            'transit_model': np.ones(1000),
            'duration': 0.1,
            'depth': 0.01,
        }
        time = np.linspace(0, 30, 1000)
        flux = np.ones_like(time)
        flux[500:510] -= 0.01  # Inject a transit
        error = np.full_like(time, 0.001)
        total_time = 30.0
        star_radius = 1.0
        cadence = 0.02
        method = 'minimize'

        fit_result = calculate_fit_for_period(
            result, time, flux, error, total_time, star_radius, cadence, method
        )
        self.assertIsNotNone(fit_result)
        self.assertIn('best_fit_params', fit_result)
        self.assertIn('final_chi2', fit_result)
        self.assertIn('best_fit_model_lightcurve', fit_result)

if __name__ == '__main__':
    unittest.main()