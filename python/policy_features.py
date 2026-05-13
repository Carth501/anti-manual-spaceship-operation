from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ThrusterConfigFeaturesExtractor(BaseFeaturesExtractor):
	def __init__(
		self,
		observation_space: spaces.Box,
		static_thruster_features: Sequence[float],
	) -> None:
		self._observation_dim = int(np.prod(observation_space.shape))
		self._static_feature_values = [float(value) for value in static_thruster_features]
		features_dim = self._observation_dim + len(self._static_feature_values)
		super().__init__(observation_space, features_dim=max(features_dim, 1))
		self.register_buffer(
			"static_thruster_features",
			th.tensor(self._static_feature_values, dtype=th.float32),
		)

	def forward(self, observations: th.Tensor) -> th.Tensor:
		observation_tensor = observations.float()
		if observation_tensor.dim() == 1:
			observation_tensor = observation_tensor.unsqueeze(0)

		flat_observations = observation_tensor.reshape(observation_tensor.shape[0], -1)
		if self.static_thruster_features.numel() == 0:
			return flat_observations

		static_features = self.static_thruster_features.unsqueeze(0).expand(flat_observations.shape[0], -1)
		return th.cat([flat_observations, static_features], dim=1)


class ThrusterSetFeaturesExtractor(BaseFeaturesExtractor):
	def __init__(
		self,
		observation_space: spaces.Box,
		global_features: Sequence[float],
		thruster_feature_rows: Sequence[Sequence[float]],
		thruster_embedding_dim: int = 32,
		global_embedding_dim: int = 16,
	) -> None:
		self._observation_dim = int(np.prod(observation_space.shape))
		self._global_feature_values = [float(value) for value in global_features]
		self._thruster_feature_rows = [
			[float(value) for value in row]
			for row in thruster_feature_rows
			if isinstance(row, Sequence)
		]
		thruster_feature_dim = len(self._thruster_feature_rows[0]) if self._thruster_feature_rows else 0
		if any(len(row) != thruster_feature_dim for row in self._thruster_feature_rows):
			raise ValueError("Thruster feature rows must all have the same width")

		self._resolved_global_embedding_dim = global_embedding_dim if self._global_feature_values else 0
		self._resolved_thruster_embedding_dim = thruster_embedding_dim if self._thruster_feature_rows else 0
		features_dim = (
			self._observation_dim
			+ self._resolved_global_embedding_dim
			+ self._resolved_thruster_embedding_dim
		)
		super().__init__(observation_space, features_dim=max(features_dim, 1))

		global_tensor = th.tensor(self._global_feature_values, dtype=th.float32)
		if self._thruster_feature_rows:
			thruster_tensor = th.tensor(self._thruster_feature_rows, dtype=th.float32)
			thruster_presence_mask = thruster_tensor[:, :1]
		else:
			thruster_tensor = th.zeros((0, 0), dtype=th.float32)
			thruster_presence_mask = th.zeros((0, 1), dtype=th.float32)
		self.register_buffer("global_feature_tensor", global_tensor)
		self.register_buffer("thruster_feature_tensor", thruster_tensor)
		self.register_buffer("thruster_presence_mask", thruster_presence_mask)

		self.global_encoder = None
		if self._resolved_global_embedding_dim > 0:
			self.global_encoder = nn.Sequential(
				nn.Linear(len(self._global_feature_values), self._resolved_global_embedding_dim),
				nn.ReLU(),
				nn.Linear(self._resolved_global_embedding_dim, self._resolved_global_embedding_dim),
				nn.ReLU(),
			)

		self.thruster_encoder = None
		if self._resolved_thruster_embedding_dim > 0 and thruster_feature_dim > 0:
			self.thruster_encoder = nn.Sequential(
				nn.Linear(thruster_feature_dim, self._resolved_thruster_embedding_dim),
				nn.ReLU(),
				nn.Linear(self._resolved_thruster_embedding_dim, self._resolved_thruster_embedding_dim),
				nn.ReLU(),
			)

	def forward(self, observations: th.Tensor) -> th.Tensor:
		observation_tensor = observations.float()
		if observation_tensor.dim() == 1:
			observation_tensor = observation_tensor.unsqueeze(0)

		flat_observations = observation_tensor.reshape(observation_tensor.shape[0], -1)
		feature_parts = [flat_observations]

		if self.global_encoder is not None and self.global_feature_tensor.numel() > 0:
			global_embedding = self.global_encoder(self.global_feature_tensor.unsqueeze(0))
			feature_parts.append(global_embedding.expand(flat_observations.shape[0], -1))

		if self.thruster_encoder is not None and self.thruster_feature_tensor.numel() > 0:
			thruster_embeddings = self.thruster_encoder(self.thruster_feature_tensor)
			if self.thruster_presence_mask.numel() > 0:
				thruster_embeddings = thruster_embeddings * self.thruster_presence_mask
				present_count = th.clamp(self.thruster_presence_mask.sum(), min=1.0)
				pooled_thruster_embedding = thruster_embeddings.sum(dim=0, keepdim=True) / present_count
			else:
				pooled_thruster_embedding = thruster_embeddings.mean(dim=0, keepdim=True)
			feature_parts.append(pooled_thruster_embedding.expand(flat_observations.shape[0], -1))

		return th.cat(feature_parts, dim=1)