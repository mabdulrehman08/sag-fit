import unittest

import numpy as np
from scipy.spatial import cKDTree

from sag_fit import (
    DEV_SEEDS, GRID, HALF_SPAN, HELDOUT_SEEDS, SCENES, catenary, clearance,
    fit_wire, run_trial, simulate, summarize,
)


class SagFitTests(unittest.TestCase):
    def test_clearance_is_a_3d_minimum(self):
        wire = np.array([[3, 4, 12], [1, 2, 2]])
        self.assertEqual(clearance(wire, cKDTree([[0, 0, 0]])), 3.0)

    def test_catenary_recovers_clean_wire_with_outliers_and_gap(self):
        x = np.linspace(-48, 48, 101)
        x = x[np.abs(x) > 10]
        wire = np.column_stack((x, 1 + 0.04 * x, catenary(x / HALF_SPAN, 2.4, 0.13, 11)))
        wire[::9, 1:] += 6
        fitted = fit_wire(wire, "catenary", seed=13)
        np.testing.assert_allclose(fitted[:, 1], 1 + 0.04 * GRID, atol=1e-5)
        np.testing.assert_allclose(fitted[:, 2], catenary(GRID / HALF_SPAN, 2.4, 0.13, 11),
                                   atol=1e-5)

    def test_sway_truth_is_worst_over_phases(self):
        static = next(scene for scene in SCENES if scene.name == "Noise 0.20, gap 12")
        sway = next(scene for scene in SCENES if scene.name == "Sway")
        still_points, still_trees, still_truth = simulate(static, seed=3)
        moving_points, moving_trees, moving_truth = simulate(sway, seed=3)
        np.testing.assert_array_equal(still_trees.data, moving_trees.data)
        np.testing.assert_array_equal(still_points[:, 0], moving_points[:, 0])
        self.assertLess(moving_truth, still_truth)

    def test_trial_reproducibility_and_split(self):
        self.assertTrue(set(DEV_SEEDS).isdisjoint(HELDOUT_SEEDS))
        trial = run_trial(SCENES[2], seed=3)
        self.assertEqual(trial, run_trial(SCENES[2], seed=3))
        stats = summarize([trial])
        for method, error in trial["signed_error_m"].items():
            self.assertAlmostEqual(stats[method]["mae_m"], abs(error))
            self.assertAlmostEqual(stats[method]["bias_m"], error)
            self.assertEqual(stats[method]["failures"], 0)


if __name__ == "__main__":
    unittest.main()
