"""Module 5 -- ASST-GCN (Adaptive Semantic-Spatio-Temporal GCN), Sheng et al. 2022, Section III.C.

Equations and mechanism below are transcribed directly from the paper's
own text.

Basic spatial GCN (Eq. 1-2), for input feature map f_in in R^(D_in x K)
(D_in = input feature dim per node, K = number of graph nodes):

    f_out = D^(-1/2) (A + I) D^(-1/2) f_in W  =  A_hat f_in W

Partition Graph Convolution (Eq. 3), splitting neighbors into Q groups,
each with its own adjacency matrix and transform:

    f_out = sum_{q=1}^{Q} A_q f_in W_q

ASST-GCN's core equation (Eq. 4) replaces each subgraph's single
adjacency matrix with the SUM of two adjacency matrices of different
character:

    f_out = sum_{q=1}^{Q} (A^se_q + A^st_q) f_in W_q

- A^se (semantic graph): "sample-independent" -- a fully-connected K x K
  matrix of entirely free, unconstrained learnable parameters (no
  structural prior at all), one A^se_q per subgraph per layer (i.e. NOT
  shared across layers), each initialized to the constant 1e-6. Fixed
  after training; does not depend on the current input at inference time.

- A^st (spatio-temporal attention graph): "sample-dependent" -- computed
  from the current input via embedding similarity (Eq. 5-6):

    s(v_i, v_j) = (W_theta v_i)^T (W_phi v_j) / sum_j (W_theta v_i)^T (W_phi v_j)   (Eq. 5)
    A^st = softmax( (W_theta f_in)^T (W_phi f_in) )                                 (Eq. 6)

  where W_theta, W_phi in R^(D_e x D_in) are learned projection matrices
  (D_e = the embedding space dimension), one pair per subgraph.

Per Fig. 3 (the ASST-GCN layer diagram): each layer's GCN unit computes
all Q subgraphs' outputs as above and CONCATENATES them (rather than
literally summing over q as Eq. 4's sigma might suggest in isolation) to
form the input to a feed-forward unit (FFN), with "multiple residual
connections" added for trainability. This implementation follows Fig. 3's
concatenation + FFN + residual structure, which is also consistent with
Eq. 4 read as "the aggregate GCN output before the FFN, expressed
per-subgraph" rather than requiring literal summation.

Values fixed elsewhere in this project (not re-derived here): 6
ASST-GCN layers are stacked, each with 8 subgraphs,
512 output channels for every layer. This implementation keeps
in_channels == out_channels == 512 throughout (matching Module 4's
semantic-embedding dimension, so no separate input projection is needed
before the first layer), making every layer's residual connections plain
identity adds.

I picked the embedding dimension D_e used for the spatio-temporal similarity 
projections (defaults to in_channels // num_subgraphs, a standard multi-head-style 
split), and the FFN's internal hidden width (defaults to 2x the model width).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from fusion_avsr.models.landmark.lrlp import NUM_LRLPS
from fusion_avsr.utils.logging import get_logger

logger = get_logger(__name__)

ASST_GCN_NUM_LAYERS = 6
ASST_GCN_NUM_SUBGRAPHS = 8
ASST_GCN_CHANNELS = 512

# Constant the paper specifies for initializing every semantic subgraph
# adjacency matrix A^se_q.
SEMANTIC_GRAPH_INIT_VALUE = 1e-6


class ASSTGCNLayer(nn.Module):
    """One Adaptive Semantic-Spatio-Temporal Graph Convolution layer (Fig. 3).

    Operates independently per frame: the K x K adjacency matrices relate
    the 38 landmarks to each other at a single time step (the paper
    states A^se, A^st in R^(K x K), with no temporal extent), so the
    graph convolution is applied with T folded into the batch dimension.
    Temporal context is instead carried by the per-node feature channels
    themselves, produced upstream by LMFE/LCFE's 5-frame receptive field.
    """

    def __init__(
        self,
        channels: int = ASST_GCN_CHANNELS,
        num_subgraphs: int = ASST_GCN_NUM_SUBGRAPHS,
        num_nodes: int = NUM_LRLPS,
        embed_dim: Optional[int] = None,
        ffn_hidden_dim: Optional[int] = None,
    ) -> None:
        """Build one ASST-GCN layer.

        Args:
            channels: Input and output channel count (identical, so
                residual connections are plain identity adds). Defaults
                to 512.
            num_subgraphs: Number of parallel subgraphs (Q). Defaults to
                8. ``channels`` must be divisible by this.
            num_nodes: Number of graph nodes (K, the 38 LRLPs).
            embed_dim: Dimension of the similarity-projection embedding
                space (D_e in Eq. 5-6) used per subgraph. Defaults to
                ``channels // num_subgraphs`` if not given.
            ffn_hidden_dim: Hidden width of the feed-forward unit.
                Defaults to ``2 * channels`` if not given.
        """
        super().__init__()
        if channels % num_subgraphs != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_subgraphs ({num_subgraphs})")

        subgraph_dim = channels // num_subgraphs
        embed_dim = embed_dim if embed_dim is not None else subgraph_dim
        ffn_hidden_dim = ffn_hidden_dim if ffn_hidden_dim is not None else 2 * channels

        self.num_subgraphs = num_subgraphs
        self.subgraph_dim = subgraph_dim

        # Sample-independent semantic graphs: one free K x K parameter
        # matrix per subgraph, initialized to the constant the paper
        # specifies.
        self.semantic_adjacency = nn.Parameter(
            torch.full((num_subgraphs, num_nodes, num_nodes), SEMANTIC_GRAPH_INIT_VALUE)
        )

        self.theta_proj = nn.ModuleList(
            [nn.Linear(channels, embed_dim, bias=False) for _ in range(num_subgraphs)]
        )
        self.phi_proj = nn.ModuleList(
            [nn.Linear(channels, embed_dim, bias=False) for _ in range(num_subgraphs)]
        )
        self.feature_transform = nn.ModuleList(
            [nn.Linear(channels, subgraph_dim, bias=False) for _ in range(num_subgraphs)]
        )

        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_hidden_dim, channels),
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """Apply one ASST-GCN layer.

        Args:
            node_features: ``(B, T, K, C)`` per-node feature tensor,
                ``K == num_nodes``, ``C == channels``.

        Returns:
            A ``(B, T, K, C)`` tensor of the same shape, after one layer
            of adaptive semantic-spatio-temporal graph convolution plus a
            position-wise feed-forward unit, both wrapped in residual
            connections.
        """
        subgraph_outputs = []
        for q in range(self.num_subgraphs):
            theta = self.theta_proj[q](node_features)  # (B, T, K, D_e)
            phi = self.phi_proj[q](node_features)  # (B, T, K, D_e)
            similarity = theta @ phi.transpose(-1, -2)  # (B, T, K, K)
            spatio_temporal_adjacency = torch.softmax(similarity, dim=-1)  # Eq. 6

            adjacency = self.semantic_adjacency[q] + spatio_temporal_adjacency  # Eq. 4, per subgraph
            aggregated = adjacency @ node_features  # (B, T, K, C), Kipf-Welling-style A_hat @ H
            subgraph_outputs.append(self.feature_transform[q](aggregated))  # (B, T, K, subgraph_dim)

        gcn_output = torch.cat(subgraph_outputs, dim=-1)  # (B, T, K, C) -- Fig. 3's concatenation
        gcn_output = node_features + gcn_output  # first residual connection

        ffn_output = self.ffn(gcn_output)
        return gcn_output + ffn_output  # second residual connection


class ASSTGCN(nn.Module):
    """Stack of 6 ASST-GCN layers, followed by global average pooling over landmarks."""

    def __init__(
        self,
        num_layers: int = ASST_GCN_NUM_LAYERS,
        channels: int = ASST_GCN_CHANNELS,
        num_subgraphs: int = ASST_GCN_NUM_SUBGRAPHS,
        num_nodes: int = NUM_LRLPS,
        output_dim: Optional[int] = None,
    ) -> None:
        """Build the ASST-GCN module.

        Args:
            num_layers: Number of stacked ASST-GCN layers. Defaults to 6.
            channels: Channel width used throughout every layer. Defaults
                to 512.
            num_subgraphs: Subgraphs per layer. Defaults to 8.
            num_nodes: Number of graph nodes (K, the 38 LRLPs).
            output_dim: If given, a learned linear projection is applied
                after global average pooling, mapping the 512-dim
                per-frame feature to this dimension -- e.g. to match the
                audio/appearance encoders' dimension for fusion. If
                ``None`` (default), no projection is applied and the raw
                512-dim pooled feature is returned.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            ASSTGCNLayer(channels=channels, num_subgraphs=num_subgraphs, num_nodes=num_nodes)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(channels, output_dim) if output_dim is not None else None
        self.channels = channels

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """Run the full ASST-GCN stack and pool over landmarks.

        Args:
            node_features: ``(B, T, K, C)`` per-node feature tensor (the
                concatenated LMFE+LCFE features plus the Module 4
                semantic embedding -- see
                ``fusion_avsr.models.landmark.encoder``).

        Returns:
            A ``(B, T, output_dim)`` tensor if ``output_dim`` was given
            at construction time, else ``(B, T, channels)``.
        """
        x = node_features
        for layer in self.layers:
            x = layer(x)

        pooled = x.mean(dim=2)  # GAP over K (landmarks), per Fig. 2(c1) -> (B, T, C)
        if self.output_projection is not None:
            pooled = self.output_projection(pooled)
        return pooled
