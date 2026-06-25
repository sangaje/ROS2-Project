from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class _FiLMTCNBlock(nn.Module):
    """One residual temporal-conv block with FiLM conditioning from the map.

    Input  : x (B, C, H)  channel-first temporal features
             map_feat (B, map_dim)
    Output : x' (B, C, H)

    The map produces (gamma, beta) which modulate the conv output:
        y = conv(x)
        y = y * (1 + gamma) + beta
        y = SiLU(y)
        x' = x + y          (residual)

    Same LiDAR-style circular padding is NOT used here because the axis is time,
    not an angular axis; zero padding over the short history window is correct.
    """

    def __init__(self, channels: int, map_dim: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.GroupNorm(num_groups=min(8, channels) if channels >= 8 else 1, num_channels=channels)
        self.film = nn.Linear(map_dim, channels * 2)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(p=float(dropout)) if dropout and dropout > 0.0 else None
        # Initialize FiLM so the block starts near identity (gamma~0, beta~0).
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, map_feat: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.norm(y)
        gamma, beta = self.film(map_feat).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        y = y * (1.0 + gamma) + beta
        y = self.act(y)
        if self.drop is not None:
            y = self.drop(y)
        return x + y


class MapVectorFeatureExtractor(BaseFeaturesExtractor):
    """
    Dict observation feature extractor.

    v27 model structure: Map-conditioned Delta-TCN (FiLM) replaces the v26 GRU.

      Per step k, the extractor first builds a learned fusion token:
        map_k         -> Map 2D CNN                  -> m_k
        lidar_k       -> LiDAR 1D CNN over bins       -> l_k
        stats_k       -> Stats MLP                    -> s_k
        concat(m_k, l_k, s_k) -> Step Fusion MLP      -> z_k

      Temporal branch (only when use_temporal_cnn=True):
        From the post-fusion feature history Z = [z_{t-H+1}, ..., z_t]
        and the delta history dZ_k = z_k - z_{k-1}, build a per-step token
        and run a Map-conditioned Delta-TCN, FiLM-modulated by the CURRENT map
        feature m_t:
            S = [ concat(z_k, dz_k) ]_{k}
            h_t = FiLM-TCN(S | m_t)
        The current-step fusion token z_t is ALSO passed straight to the final
        fusion as a skip connection, so the policy/critic never lose direct
        access to the current observation even if the temporal branch is noisy.

      Final:
        concat(h_t, z_t) -> final fusion -> SAC Actor / Twin Critic
        (non-temporal: z_t -> final fusion)

    Why this is better than the GRU version:
      - No recurrent hidden state: matches SAC replay-buffer re-encoding.
      - Current step has a direct skip path to the head (no GRU bottleneck).
      - The TCN reads recent *dynamics* (deltas), not a long memory.
      - The map conditions the temporal interpretation via FiLM: the same LiDAR
        change means different things in a corridor vs an open room.

    Interface is unchanged from v26 so train_sac.py and SB3 SAC need no edits:
      __init__(observation_space, map_features_dim, vector_features_dim,
               temporal_features_dim, combined_features_dim, use_temporal_cnn,
               lidar_dim)
      forward(observations_dict) -> (B, combined_features_dim)

    Expected observation:
      map     : (5, 32, 32)
      vector  : (vector_dim,) = lidar(L) + stats(S)
      map_seq : (H, 5, 32, 32)   [only when use_temporal_cnn]
      seq     : (H, vector_dim)  [only when use_temporal_cnn]
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
            # v27: Map-conditioned Delta-TCN over post-fusion features.
            #
            # Per-step temporal token is [z_k, dz_k] where dz_k = z_k - z_{k-1}.
            # This gives the TCN both the current learned step feature and its
            # delta, matching the "recent dynamics detector" role from the design
            # note (delta/action sequence rather than long memory).
            tcn_in_dim = vector_features_dim * 2  # concat(z_k, dz_k)
            tcn_hidden = temporal_features_dim

            self.temporal_input_proj = nn.Sequential(
                nn.Linear(tcn_in_dim, tcn_hidden),
                nn.LayerNorm(tcn_hidden),
                nn.SiLU(),
            )
            # FiLM is conditioned on the CURRENT map feature m_t.
            self.temporal_block1 = _FiLMTCNBlock(tcn_hidden, map_features_dim, kernel_size=3, dropout=0.0)
            self.temporal_block2 = _FiLMTCNBlock(tcn_hidden, map_features_dim, kernel_size=3, dropout=0.0)
            self.temporal_pool = nn.AdaptiveAvgPool1d(1)
            self.temporal_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(tcn_hidden, temporal_features_dim),
                nn.LayerNorm(temporal_features_dim),
                nn.SiLU(),
            )
            # Kept as None so any old introspection that referenced these does not
            # mistake the TCN for a GRU / raw-sequence Conv1D temporal block.
            self.temporal_gru = None
            self.temporal_cnn = None
            # Final input: temporal summary h_t  +  current-step skip token z_t.
            final_in_dim = temporal_features_dim + vector_features_dim
        else:
            self.temporal_input_proj = None
            self.temporal_block1 = None
            self.temporal_block2 = None
            self.temporal_pool = None
            self.temporal_head = None
            self.temporal_gru = None
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

    # ------------------------------------------------------------------ #
    # Sanitizers
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Per-step encoders
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Temporal branch: Map-conditioned Delta-TCN with FiLM
    # ------------------------------------------------------------------ #
    def _encode_temporal_map_conditioned_delta_tcn(
        self, map_seq: torch.Tensor, seq_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (h_t temporal summary, z_t current-step skip token).

        Raw map_seq/seq are NOT fed to the temporal module directly. Each history
        item first becomes z_k via the same learned step encoders, then the TCN
        operates on [z_k, dz_k] with FiLM conditioning from the current map m_t.
        """
        if self.temporal_input_proj is None:
            raise RuntimeError("Delta-TCN requested but temporal modules are None")

        map_seq = self._sanitize_map(map_seq)
        seq_obs = self._sanitize_vector(seq_obs)

        batch_size = int(seq_obs.shape[0])
        history_len = int(seq_obs.shape[1])
        flat_maps = map_seq.reshape(batch_size * history_len, *map_seq.shape[2:])
        flat_vecs = seq_obs.reshape(batch_size * history_len, -1)

        # z_k for every history step.
        z_flat = self._encode_full_step_flat(flat_maps, flat_vecs)
        z_seq = z_flat.reshape(batch_size, history_len, self.vector_features_dim)  # (B,H,D)

        # Current-step token z_t (last in history) for the skip connection and
        # for FiLM conditioning via its map feature.
        z_t = z_seq[:, -1, :]  # (B, D)

        # Delta over time: dz_k = z_k - z_{k-1}, with dz_0 = 0.
        dz_seq = torch.zeros_like(z_seq)
        if history_len > 1:
            dz_seq[:, 1:, :] = z_seq[:, 1:, :] - z_seq[:, :-1, :]

        # Per-step temporal token = concat(z_k, dz_k) -> (B,H,2D).
        s_seq = torch.cat([z_seq, dz_seq], dim=-1)
        s_proj = self.temporal_input_proj(s_seq)        # (B,H,hidden)
        x = s_proj.transpose(1, 2).contiguous()         # (B,hidden,H)

        # FiLM conditioning uses the CURRENT map feature m_t.
        m_t = self._encode_map_flat(map_seq[:, -1])     # (B, map_features_dim)

        x = self.temporal_block1(x, m_t)
        x = self.temporal_block2(x, m_t)
        x = self.temporal_pool(x)                       # (B,hidden,1)
        h_t = self.temporal_head(x)                     # (B, temporal_features_dim)
        return h_t, z_t

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.use_temporal_cnn and "map_seq" in observations and "seq" in observations:
            h_t, z_t = self._encode_temporal_map_conditioned_delta_tcn(
                observations["map_seq"], observations["seq"]
            )
            features = torch.cat([h_t, z_t], dim=1)
        else:
            features = self._encode_full_step_flat(observations["map"], observations["vector"])
        return self.final_fusion(features)
