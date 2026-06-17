from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class MapVectorFeatureExtractor(BaseFeaturesExtractor):
    """
    Dict observation 전용 feature extractor.

    observation_space:
      {
        "map":    Box(0, 1, shape=(5, H, W), dtype=float32),
        "vector": Box(-1, 1, shape=(N,), dtype=float32),
        "seq":    Box(-1, 1, shape=(T, N), dtype=float32)  # optional
      }

    v17 stabilization:
      - BatchNorm은 사용하지 않는다. RL에서는 train mini-batch 통계와
        action-selection 시점의 단일 observation 통계가 달라져 제어가 튈 수 있다.
      - CNN branch에는 GroupNorm을 사용한다.
      - MLP/vector/fusion branch에는 LayerNorm을 사용한다.
      - 입력 NaN/Inf를 제거하고, map/lidar/scalar 범위를 다시 clamp한다.

    최신 구조:
      - map branch:
          5-channel robot-centric map을 약한 2D CNN + Global Average Pooling으로 인코딩.
          channel 0..2 = SLAM free/unknown/occupied one-hot
          channel 3    = confidence
          channel 4    = priority

      - vector branch:
          vector의 앞 360개 LiDAR bin은 항상 1D CNN으로 각도 방향 local pattern을 인코딩.
          나머지 scalar stats는 MLP로 인코딩.
          [lidar_feature, stats_feature]를 concat한 뒤 vector feature로 projection.

      - temporal branch:
          raw seq(B,T,370)를 그대로 Conv1d에 넣지 않는다.
          각 time step마다 같은 LiDAR 1D CNN + stats MLP로 token을 만든다.
          token sequence (B,T,D)를 temporal 1D CNN으로 처리한다.
    """

    LIDAR_DIM = 360

    def __init__(
        self,
        observation_space: spaces.Dict,
        map_features_dim: int = 128,
        vector_features_dim: int = 128,
        temporal_features_dim: int = 128,
        combined_features_dim: int = 256,
        use_temporal_cnn: bool = True,
    ):
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError(
                "MapVectorFeatureExtractor requires a spaces.Dict observation space"
            )

        super().__init__(observation_space, features_dim=int(combined_features_dim))

        map_space = observation_space.spaces["map"]
        vector_space = observation_space.spaces["vector"]

        map_channels = int(map_space.shape[0])
        vector_dim = int(vector_space.shape[0])

        if vector_dim < self.LIDAR_DIM:
            raise ValueError(
                f"vector observation dim must be >= {self.LIDAR_DIM}, got {vector_dim}"
            )

        self.vector_dim = vector_dim
        self.lidar_dim = self.LIDAR_DIM
        self.stats_dim = vector_dim - self.LIDAR_DIM
        self.use_temporal_cnn = bool(
            use_temporal_cnn and "seq" in observation_space.spaces
        )

        vector_features_dim = int(vector_features_dim)
        map_features_dim = int(map_features_dim)
        temporal_features_dim = int(temporal_features_dim)
        combined_features_dim = int(combined_features_dim)

        # 기존 CLI 인자 수를 늘리지 않기 위해 vector_features_dim 내부에서
        # LiDAR feature와 scalar stats feature 차원을 나눈다.
        self.lidar_features_dim = max(32, int(round(vector_features_dim * 0.65)))
        self.stats_features_dim = max(16, vector_features_dim - self.lidar_features_dim)
        if self.stats_dim == 0:
            self.stats_features_dim = 0

        token_dim = self.lidar_features_dim + self.stats_features_dim
        self.temporal_token_dim = token_dim

        # ------------------------------------------------------------------
        # 1) Map branch: weak 2D CNN + GroupNorm + Global Average Pooling
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
        # 2) LiDAR branch: 360-bin circular 1D signal encoder + GroupNorm
        # ------------------------------------------------------------------
        self.lidar_cnn = nn.Sequential(
            nn.Conv1d(
                1, 16, kernel_size=5, stride=2, padding=3, padding_mode="circular"
            ),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.SiLU(),
            nn.Conv1d(
                16, 32, kernel_size=5, stride=2, padding=2, padding_mode="circular"
            ),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.SiLU(),
            nn.Conv1d(
                32, 64, kernel_size=5, stride=2, padding=2, padding_mode="circular"
            ),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, self.lidar_features_dim),
            nn.LayerNorm(self.lidar_features_dim),
            nn.SiLU(),
        )

        # ------------------------------------------------------------------
        # 3) Scalar stats branch: non-LiDAR feature encoder + LayerNorm
        # ------------------------------------------------------------------
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

        # Current vector projection.
        # 입력은 [lidar_feature, stats_feature].
        self.vector_net = nn.Sequential(
            nn.Linear(token_dim, vector_features_dim),
            nn.LayerNorm(vector_features_dim),
            nn.SiLU(),
            nn.Linear(vector_features_dim, vector_features_dim),
            nn.LayerNorm(vector_features_dim),
            nn.SiLU(),
        )

        fusion_in_dim = map_features_dim + vector_features_dim

        # ------------------------------------------------------------------
        # 4) Temporal branch: encoded token sequence -> 1D CNN over time
        # ------------------------------------------------------------------
        if self.use_temporal_cnn:
            seq_space = observation_space.spaces["seq"]
            history_len = int(seq_space.shape[0])
            seq_vector_dim = int(seq_space.shape[1])

            if seq_vector_dim != vector_dim:
                raise ValueError(
                    f"seq vector dim must match vector dim: seq={seq_vector_dim}, vector={vector_dim}"
                )

            self.history_len = history_len
            self.temporal_cnn = nn.Sequential(
                nn.Conv1d(token_dim, 128, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=128),
                nn.SiLU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=128),
                nn.SiLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(128, temporal_features_dim),
                nn.LayerNorm(temporal_features_dim),
                nn.SiLU(),
            )
            fusion_in_dim += temporal_features_dim
        else:
            self.history_len = 0
            self.temporal_cnn = None

        # ------------------------------------------------------------------
        # 5) Final fusion
        # ------------------------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, combined_features_dim),
            nn.LayerNorm(combined_features_dim),
            nn.SiLU(),
        )

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

    def _split_vector(
        self, vector_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        vector_obs = self._sanitize_vector(vector_obs)
        lidar = vector_obs[..., : self.lidar_dim]
        stats = vector_obs[..., self.lidar_dim :] if self.stats_dim > 0 else None
        return lidar, stats

    def _encode_lidar_flat(self, lidar_flat: torch.Tensor) -> torch.Tensor:
        # lidar_flat: (B, 360), already normalized to 0..1.
        lidar_flat = torch.clamp(lidar_flat.float(), 0.0, 1.0).unsqueeze(1)  # (B, 1, 360)
        return self.lidar_cnn(lidar_flat)

    def _encode_stats_flat(
        self, stats_flat: torch.Tensor | None, batch_size: int
    ) -> torch.Tensor | None:
        if self.stats_net is None or stats_flat is None:
            return None
        stats_flat = torch.nan_to_num(stats_flat.float(), nan=0.0, posinf=1.0, neginf=-1.0)
        stats_flat = torch.clamp(stats_flat, -1.0, 1.0)
        return self.stats_net(stats_flat)

    def _encode_vector_token_flat(self, vector_flat: torch.Tensor) -> torch.Tensor:
        # vector_flat: (B, 370)
        vector_flat = self._sanitize_vector(vector_flat)
        lidar, stats = self._split_vector(vector_flat)
        lidar_features = self._encode_lidar_flat(lidar)
        stats_features = self._encode_stats_flat(stats, vector_flat.shape[0])

        if stats_features is None:
            return lidar_features

        return torch.cat([lidar_features, stats_features], dim=1)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        map_obs = self._sanitize_map(observations["map"])
        vector_obs = self._sanitize_vector(observations["vector"])

        map_features = self.map_linear(self.map_cnn(map_obs))

        current_token = self._encode_vector_token_flat(vector_obs)
        vector_features = self.vector_net(current_token)

        features = [map_features, vector_features]

        if self.use_temporal_cnn and self.temporal_cnn is not None:
            seq_obs = observations["seq"].float()  # (B, T, N)
            seq_obs = torch.nan_to_num(seq_obs, nan=0.0, posinf=1.0, neginf=-1.0)
            batch_size, history_len, vector_dim = seq_obs.shape

            # 각 time step의 vector를 같은 LiDAR CNN + stats MLP로 token화한다.
            seq_flat = seq_obs.reshape(batch_size * history_len, vector_dim)
            seq_token_flat = self._encode_vector_token_flat(seq_flat)
            seq_tokens = seq_token_flat.reshape(
                batch_size,
                history_len,
                self.temporal_token_dim,
            )

            # Conv1d expects (B, C=token_dim, L=T).
            seq_tokens = seq_tokens.transpose(1, 2)
            temporal_features = self.temporal_cnn(seq_tokens)
            features.append(temporal_features)

        return self.fusion(torch.cat(features, dim=1))
