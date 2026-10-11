import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import metrics


class MetricTests(unittest.TestCase):
    def test_hand_calculation(self):
        r = metrics([[0.8, 0.2], [0.3, 0.7]], [0, 0])
        self.assertAlmostEqual(r["accuracy"], 50)
        self.assertAlmostEqual(r["nll"], -(np.log(.8)+np.log(.3))/2)
        self.assertAlmostEqual(r["brier"], (.08+.98)/2)
        self.assertAlmostEqual(r["ece"], 45)

    def test_perfect_and_reject_bad_probabilities(self):
        r = metrics(np.eye(2), [0, 1])
        self.assertEqual(r["ece"], 0)
        self.assertEqual(r["nll"], 0)
        with self.assertRaises(ValueError):
            metrics([[1.1, -.1]], [0])


if __name__ == "__main__":
    unittest.main()
