from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class MapVectorFeatureExtractor(BaseFeaturesExtractor):
    """
    Dict observation feature extractor.

    v26 model structure: GRU receives ONLY post-fusion features.

      Per step k, the extractor first builds a learned fusion token:
        map_k         -> Map 2D CNN                  -> m_k
        lidar_k       -> LiDAR 1D CNN over 60 bins    -> l_k
        stats_k       -> Stats MLP                    -> s_k
        concat(m_k, l_k, s_k) -> Step Fusion MLP      -> z_k

      Temporal branch:
        Z = [z_{t-H+1}, ..., z_t]
        Z only -> GRU over time -> q_t

      Final:
        q_t -> final fusion -> SAC Actor / Twin Critic

    Important terminology:
      - LiDAR 1D CNN is NOT temporal. It is Conv1D over the LiDAR bin/angular axis only.
      - GRU never sees raw map_seq, raw LiDAR, or raw stats directly.
        It receives only the post-fusion feature sequence Z.
      - map_seq and seq still exist in the observation because SAC replay-buffer training
        needs the raw history snapshots. They are used only to reconstruct z_k with the
        current learned encoders before the GRU.

    Expected observation with 60-bin LiDAR:
      map     : (5, 32, 32)
      vector  : (70,) = lidar(60) + stats(10)
      map_seq : (H, 5, 32, 32)
      seq     : (H, 70)
    """

    DEFAULT_LIDAR_DIM = 360

    def __init__(
        self,
        observation_space: spaces.Dict,
        map_features_dim: int = 128,
        vector_features_dim: int = 128,
        temporal_features_dim: int = 128,
        combined_features_dim: int = 256,
        use_temporal_cnn: bool = False,
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

        self.use_temporal_cnn = bool(
            use_temporal_cnn
            and ("seq" in observation_space.spaces)
            and ("map_seq" in observation_space.spaces)
        )
        self.history_len = 0
        if self.use_temporal_cnn:
            seq_space = observation_space.spaces["seq"]
            map_seq_space = observation_space.spaces["map_seq"]
            if len(seq_space.shape) != 2:
                raise ValueError(f"obs['seq'] must have shape (H, vector_dim), got {seq_space.shape}")
            if len(map_seq_space.shape) != 4:
                raise ValueError(f"obs['map_seq'] must have shape (H,C,H,W), got {map_seq_space.shape}")
            self.history_len = int(seq_space.shape[0])
            if int(map_seq_space.shape[0]) != self.history_len:
                raise ValueError(
                    f"seq history length {seq_space.shape[0]} != map_seq history length {map_seq_space.shape[0]}"
                )

        vector_features_dim = int(vector_features_dim)
        map_features_dim = int(map_features_dim)
        temporal_features_dim = int(temporal_features_dim)
        combined_features_dim = int(combined_features_dim)

        self.map_features_dim = map_features_dim
        self.vector_features_dim = vector_features_dim
        self.temporal_features_dim = temporal_features_dim if self.use_temporal_cnn else 0
        self.combined_features_dim = combined_features_dim

        # Split vector branch: LiDAR uses Conv1D over angular bins, stats use MLP only.
        self.lidar_features_dim = max(32, int(round(vector_features_dim * 0.65)))
        self.stats_features_dim = max(16, vector_features_dim - self.lidar_features_dim)
        if self.stats_dim == 0:
            self.stats_features_dim = 0
        vector_token_dim = self.lidar_features_dim + self.stats_features_dim
        full_step_token_dim = map_features_dim + vector_token_dim

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

        # LiDAR angular/range 1D CNN. This Conv1D is over the LiDAR bin axis only.
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

        # FULL per-step fusion: map feature + lidar feature + stats feature -> z_k.
        self.step_fusion = nn.Sequential(
            nn.Linear(full_step_token_dim, vector_features_dim),
            nn.LayerNorm(vector_features_dim),
            nn.SiLU(),
            nn.Linear(vector_features_dim, vector_features_dim),
            nn.LayerNorm(vector_features_dim),
            nn.SiLU(),
        )

        if self.use_temporal_cnn:
            # v26: recurrent temporal aggregator over post-fusion features only.
            # Input shape at runtime: (B, H, vector_features_dim), where each z_k is
            # Fusion(MapCNN(map_k), LiDARCNN(lidar_k), StatsMLP(stats_k)).
            self.temporal_gru = nn.GRU(
                input_size=vector_features_dim,
                hidden_size=temporal_features_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
            self.temporal_head = nn.Sequential(
                nn.LayerNorm(temporal_features_dim),
                nn.SiLU(),
                nn.Linear(temporal_features_dim, temporal_features_dim),
                nn.LayerNorm(temporal_features_dim),
                nn.SiLU(),
            )
            # Keep the old attribute name as None so old debug/introspection code does not
            # accidentally treat the GRU as a raw-sequence Conv1D temporal block.
            self.temporal_cnn = None
            final_in_dim = temporal_features_dim
        else:
            self.temporal_gru = None
            self.temporal_head = None
            self.temporal_cnn = None
            final_in_dim = vector_features_dim

        self.final_fusion = nn.Sequential(
            nn.Linear(final_in_dim, combined_features_dim),
            nn.LayerNorm(combined_features_dim),
            nn.SiLU(),
            nn.Linear(combined_features_dim, combined_features_dim),
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

    def _encode_full_step_flat(self, map_flat: torch.Tensor, vector_flat: torch.Tensor) -> torch.Tensor:
        map_features = self._encode_map_flat(map_flat)
        vector_token = self._encode_vector_token_flat(vector_flat)
        return self.step_fusion(torch.cat([map_features, vector_token], dim=1))

    def _encode_temporal_from_fusion_feature_history(self, map_seq: torch.Tensor, seq_obs: torch.Tensor) -> torch.Tensor:
        if self.temporal_gru is None or self.temporal_head is None:
            raise RuntimeError("Fusion-feature GRU requested but temporal_gru is None")
        map_seq = self._sanitize_map(map_seq)
        seq_obs = self._sanitize_vector(seq_obs)
        batch_size = int(seq_obs.shape[0])
        history_len = int(seq_obs.shape[1])
        flat_maps = map_seq.reshape(batch_size * history_len, *map_seq.shape[2:])
        flat_vecs = seq_obs.reshape(batch_size * history_len, -1)
        # Build the learned post-fusion feature history first.
        # Raw map_seq/seq are NOT fed to the temporal module directly.
        # Each history item becomes z_k = Fusion(MapCNN(map_k), LiDARCNN(lidar_k), StatsMLP(stats_k)).
        fusion_features = self._encode_full_step_flat(flat_maps, flat_vecs)
        fusion_features = fusion_features.reshape(batch_size, history_len, self.vector_features_dim)
        # GRU over time after fusion only: (B,H,D) -> last hidden state (B, temporal_features_dim).
        _, h_n = self.temporal_gru(fusion_features.contiguous())
        last_hidden = h_n[-1]
        return self.temporal_head(last_hidden)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.use_temporal_cnn and "map_seq" in observations and "seq" in observations:
            features = self._encode_temporal_from_fusion_feature_history(observations["map_seq"], observations["seq"])
        else:
            features = self._encode_full_step_flat(observations["map"], observations["vector"])
        return self.final_fusion(features)
