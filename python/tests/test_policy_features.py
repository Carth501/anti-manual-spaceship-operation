from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch as th
from gymnasium import spaces


PROJECT_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_PYTHON_DIR) not in sys.path:
	sys.path.insert(0, str(PROJECT_PYTHON_DIR))

from policy_features import ThrusterSetFeaturesExtractor  # noqa: E402


class PolicyFeatureExtractorTests(unittest.TestCase):
	def test_thruster_set_extractor_outputs_expected_shape(self) -> None:
		observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
		extractor = ThrusterSetFeaturesExtractor(
			observation_space,
			global_features=[0.5, 1.0, 2.0],
			thruster_feature_rows=[
				[1.0, 1.0, 0.1, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
				[1.0, 0.0, -0.1, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.5, 0.5, 0.5],
			],
			thruster_embedding_dim=8,
			global_embedding_dim=4,
		)

		encoded = extractor(th.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0, 0.0]], dtype=th.float32))

		self.assertEqual(encoded.shape, (2, 17))


if __name__ == "__main__":
	unittest.main()