from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class MapVectorFeatureExtractor(BaseFeaturesExtractor):
    """
    Dict observation feature extractor.

    v26 model structure requested by user:

      For each history step k:
        map_k    -> shared Map 2D CNN
        vector_k -> shared LiDAR circular 1D CNN + Stats MLP
        concat(map_feature_k, vector_feature_k)
        -> shared per-step Feature Fusion MLP -> z_k

      Then:
        Z = [z_{t-H+1}, ..., z_t]
        -> Temporal 1D CNN over the time axis
        -> final feature for SAC Actor / Twin Critic

    Important distinction from v25:
      v25 applied temporal CNN separately to map_seq and vector_seq before final fusion.
      v26 applies temporal CNN to the already-fused per-step feature sequence.

    Expected v26 observation with 60-bin LiDAR and temporal history 8:
      map     : (5, 48, 48)
      vector  : (70,)
      seq     : (8, 70)
      map_seq : (8, 5, 48, 48)
    """

    DEFAULT_LIDAR_DIM = 360

    def __init__(
        self,
        observation_space: spaces.Dict,
        map_features_dim: int = 128,
        vector_features_dim: int = 128,
        temporal_features_dim: int = 128,
        combined_features_dim: int = 256,
        use_temporal_cnn: bool = True,
        lidar_dim: int | None = None,
    ):
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("MapVectorFeatureExtractor requires a spaces.Dict observation space")

        super().__init__(observation_space, features_dim=int(combined_features_dim))

        map_space = observation_space.spaces["map"]
        vector_space = observation_space.spaces["vector"]
        map_channels = int(map_space.shape[0])
        vector_dim = int(vector_space.shape[0])

        if lidar_dim is None:
            inferred = int(vector_dim) - 10
            if inferred <= 0:
                inferred = min(int(vector_dim), int(self.DEFAULT_LIDAR_DIM))
            lidar_dim = inferred

        self.vector_dim = int(vector_dim)
        self.lidar_dim = max(1, min(int(lidar_dim), int(vector_dim)))
        self.stats_dim = int(vector_dim) - int(self.lidar_dim)
        self.use_temporal_cnn = bool(use_temporal_cnn and "seq" in observation_space.spaces)
        self.temporal_uses_map_seq = bool(self.use_temporal_cnn and "map_seq" in observation_space.spaces)

        vector_features_dim = int(vector_features_dim)
        map_features_dim = int(map_features_dim)
        temporal_features_dim = int(temporal_features_dim)
        combined_features_dim = int(combined_features_dim)

        self.map_features_dim = map_features_dim
        self.vector_features_dim = vector_features_dim
        self.temporal_features_dim = temporal_features_dim
        self.combined_features_dim = combined_features_dim

        self.lidar_features_dim = max(32, int(round(vector_features_dim * 0.65)))
        self.stats_features_dim = max(16, vector_features_dim - self.lidar_features_dim)
        if self.stats_dim == 0:
            self.stats_features_dim = 0

        vector_token_dim = self.lidar_features_dim + self.stats_features_dim

        # ------------------------------------------------------------------
        # Shared current/history map encoder.
        # ------------------------------------------------------------------
        self.map_cnn = nn.Sequential(
            nn.Conv2d(map_channels, 8, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(num_groups=4, num_channels=8),
            nn.SiLU(),
            nn.Dropout2d(p=0.03),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.SiLU(),
            nn.Dropout2d(p=0.03),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, *map_space.shape, dtype=torch.float32)
            cnn_flatten_dim = int(self.map_cnn(sample).shape[1])
        self.map_linear = nn.Sequential(
            nn.Linear(cnn_flatten_dim, map_features_dim),
            nn.LayerNorm(map_features_dim),
            nn.SiLU(),
        )

        # ------------------------------------------------------------------
        # Shared current/history vector encoder.
        # LiDAR keeps coarse angular layout using AdaptiveAvgPool1d(4), instead
        # of collapsing all angle locations into one scalar token.
        # ------------------------------------------------------------------
        self.lidar_pool_bins = 4
        self.lidar_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2, padding_mode="circular"),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, padding_mode="circular"),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, padding_mode="circular"),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(self.lidar_pool_bins),
            nn.Flatten(),
        )
        self.lidar_linear = nn.Sequential(
            nn.Linear(64 * self.lidar_pool_bins, self.lidar_features_dim),
            nn.LayerNorm(self.lidar_features_dim),
            nn.SiLU(),
        )

        if self.stats_dim > 0:
            self.stats_net = nn.Sequential(
                nn.Linear(self.stats_dim, self.stats_features_dim),
                nn.LayerNorm(self.stats_features_dim),
                nn.SiLU(),
                nn.Linear(self.stats_features_dim, self.stats_features_dim),
                nn.LayerNorm(self.stats_features_dim),
                nn.SiLU(),
            )
        else:
            self.stats_net = None

        self.vector_net = nn.Sequential(
            nn.Linear(vector_token_dim, vector_features_dim),
            nn.LayerNorm(vector_features_dim),
            nn.SiLU(),
            nn.Linear(vector_features_dim, vector_features_dim),
            nn.LayerNorm(vector_features_dim),
            nn.SiLU(),
        )

        # ------------------------------------------------------------------
        # Shared per-step Feature Fusion.
        # This is the z_k encoder that is applied to both current observation
        # and each historical observation.
        # ------------------------------------------------------------------
        per_step_in_dim = map_features_dim + vector_features_dim
        self.fusion_token_dim = combined_features_dim
        self.per_step_fusion = nn.Sequential(
            nn.Linear(per_step_in_dim, combined_features_dim),
            nn.LayerNorm(combined_features_dim),
            nn.SiLU(),
            nn.Linear(combined_features_dim, combined_features_dim),
            nn.LayerNorm(combined_features_dim),
            nn.SiLU(),
        )

        # ------------------------------------------------------------------
        # Temporal CNN over fused feature sequence Z = [z_{t-H+1}, ..., z_t].
        # Input layout for Conv1D is (B, D, H), so convolution happens along
        # the H time axis, not the feature-channel axis.
        # ------------------------------------------------------------------
        if self.use_temporal_cnn:
            seq_space = observation_space.spaces["seq"]
            self.history_len = int(seq_space.shape[0])
            seq_vector_dim = int(seq_space.shape[1])
            if seq_vector_dim != vector_dim:
                raise ValueError(f"seq vector dim must match vector dim: seq={seq_vector_dim}, vector={vector_dim}")

            if not self.temporal_uses_map_seq:
                raise ValueError(
                    "v26 fused-feature temporal CNN requires map_seq in the observation space. "
                    "Check that the environment is built with map_seq history enabled."
                )

            map_seq_space = observation_space.spaces["map_seq"]
            if int(map_seq_space.shape[0]) != self.history_len:
                raise ValueError("map_seq history length must match seq history length")
            if int(map_seq_space.shape[1]) != map_channels:
                raise ValueError("map_seq channel count must match map channel count")

            mid_dim = max(64, int(temporal_features_dim) * 2)
            self.fused_temporal_cnn = nn.Sequential(
                nn.Conv1d(combined_features_dim, mid_dim, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=mid_dim),
                nn.SiLU(),
                nn.Conv1d(mid_dim, mid_dim, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=mid_dim),
                nn.SiLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(mid_dim, combined_features_dim),
                nn.LayerNorm(combined_features_dim),
                nn.SiLU(),
            )
        else:
            self.history_len = 0
            self.fused_temporal_cnn = None

    def _sanitize_map(self, map_obs: torch.Tensor) -> torch.Tensor:
        map_obs = torch.nan_to_num(map_obs.float(), nan=0.0, posinf=1.0, neginf=0.0)
        return torch.clamp(map_obs, 0.0, 1.0)

    def _sanitize_vector(self, vector_obs: torch.Tensor) -> torch.Tensor:
        vector_obs = torch.nan_to_num(vector_obs.float(), nan=0.0, posinf=1.0, neginf=-1.0)
        if vector_obs.shape[-1] >= self.lidar_dim:
            lidar = torch.clamp(vector_obs[..., : self.lidar_dim], 0.0, 1.0)
            stats = torch.clamp(vector_obs[..., self.lidar_dim :], -1.0, 1.0)
            return torch.cat([lidar, stats], dim=-1)
        return torch.clamp(vector_obs, -1.0, 1.0)

    def _split_vector(self, vector_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        vector_obs = self._sanitize_vector(vector_obs)
        lidar = vector_obs[..., : self.lidar_dim]
        stats = vector_obs[..., self.lidar_dim :] if self.stats_dim > 0 else None
        return lidar, stats

    def _encode_map_flat(self, map_flat: torch.Tensor) -> torch.Tensor:
        return self.map_linear(self.map_cnn(self._sanitize_map(map_flat)))

    def _encode_lidar_flat(self, lidar_flat: torch.Tensor) -> torch.Tensor:
        lidar_flat = torch.clamp(lidar_flat.float(), 0.0, 1.0).unsqueeze(1)
        return self.lidar_linear(self.lidar_conv(lidar_flat))

    def _encode_stats_flat(self, stats_flat: torch.Tensor | None) -> torch.Tensor | None:
        if self.stats_net is None or stats_flat is None:
            return None
        stats_flat = torch.nan_to_num(stats_flat.float(), nan=0.0, posinf=1.0, neginf=-1.0)
        stats_flat = torch.clamp(stats_flat, -1.0, 1.0)
        return self.stats_net(stats_flat)

    def _encode_vector_token_flat(self, vector_flat: torch.Tensor) -> torch.Tensor:
        vector_flat = self._sanitize_vector(vector_flat)
        lidar, stats = self._split_vector(vector_flat)
        lidar_features = self._encode_lidar_flat(lidar)
        stats_features = self._encode_stats_flat(stats)
        if stats_features is None:
            return lidar_features
        return torch.cat([lidar_features, stats_features], dim=1)

    def _encode_vector_feature_flat(self, vector_flat: torch.Tensor) -> torch.Tensor:
        return self.vector_net(self._encode_vector_token_flat(vector_flat))

    def _fuse_flat(self, map_features: torch.Tensor, vector_features: torch.Tensor) -> torch.Tensor:
        return self.per_step_fusion(torch.cat([map_features, vector_features], dim=1))

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        map_obs = self._sanitize_map(observations["map"])
        vector_obs = self._sanitize_vector(observations["vector"])

        current_map_features = self._encode_map_flat(map_obs)
        current_vector_features = self._encode_vector_feature_flat(vector_obs)
        current_z = self._fuse_flat(current_map_features, current_vector_features)

        if not self.use_temporal_cnn or self.fused_temporal_cnn is None:
            return current_z

        seq_obs = observations["seq"].float()
        seq_obs = torch.nan_to_num(seq_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        map_seq_obs = observations["map_seq"].float()
        map_seq_obs = torch.nan_to_num(map_seq_obs, nan=0.0, posinf=1.0, neginf=0.0)

        batch_size, history_len, vector_dim = seq_obs.shape
        _, map_history_len, c, h, w = map_seq_obs.shape
        if int(map_history_len) != int(history_len):
            raise RuntimeError("map_seq and seq history lengths differ")

        seq_flat = seq_obs.reshape(batch_size * history_len, vector_dim)
        map_seq_flat = map_seq_obs.reshape(batch_size * history_len, c, h, w)

        seq_map_features = self._encode_map_flat(map_seq_flat)
        seq_vector_features = self._encode_vector_feature_flat(seq_flat)
        fused_seq = self._fuse_flat(seq_map_features, seq_vector_features).reshape(
            batch_size, history_len, self.fusion_token_dim
        )

        # Guard against an incorrectly initialized first reset where history may
        # not yet contain the current frame. This keeps the last temporal token
        # exactly equal to the current observation's fused feature.
        fused_seq = fused_seq.clone()
        fused_seq[:, -1, :] = current_z

        # Conv1D over time axis: (B, H, D) -> (B, D, H)
        temporal_input = fused_seq.transpose(1, 2)
        return self.fused_temporal_cnn(temporal_input)
