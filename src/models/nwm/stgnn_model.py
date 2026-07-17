"""
Spatio-Temporal Graph Neural Network for the Neural Weather Model (NWM).
Based on Encoder-Processor-Decoder architecture over a spherical mesh.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, LayerNorm


def _batch_edges(
    edge_idx: torch.Tensor,
    num_src: int,
    num_tgt: int,
    batch_size: int,
) -> torch.Tensor:
    """Vectorised batch-offset for a single-graph edge index.

    Converts a [2, E] per-graph edge index into a [2, B*E] batched edge
    index by offsetting source and target indices for each element in the
    batch.  This avoids a Python-level loop over the batch dimension.

    Args:
        edge_idx:   [2, E] edge index (source row 0, target row 1).
        num_src:    Number of source nodes per graph.
        num_tgt:    Number of target nodes per graph.
        batch_size: Batch size B.

    Returns:
        [2, B*E] batched edge index.
    """
    device = edge_idx.device
    num_edges = edge_idx.size(1)

    # Offsets: [B] → [B*E] for source and target
    src_offsets = (
        torch.arange(batch_size, device=device) * num_src
    ).repeat_interleave(num_edges)
    tgt_offsets = (
        torch.arange(batch_size, device=device) * num_tgt
    ).repeat_interleave(num_edges)

    src = edge_idx[0].repeat(batch_size) + src_offsets
    tgt = edge_idx[1].repeat(batch_size) + tgt_offsets
    return torch.stack([src, tgt], dim=0)


class BipartiteGNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = 0):
        super().__init__()
        # PyG SAGEConv supports bipartite graphs if passed as tuple
        self.conv = SAGEConv((in_channels, in_channels), out_channels)
        self.norm = LayerNorm(out_channels)
        self.act = nn.GELU()
        
    def forward(self, x_source: torch.Tensor, x_target: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x_source: features of source nodes
        x_target: features of target nodes
        edge_index: [2, num_edges] where row 0 is source and row 1 is target
        """
        out = self.conv((x_source, x_target), edge_index)
        out = self.norm(out)
        out = self.act(out)
        return out


class NWMModel(nn.Module):
    def __init__(
        self, 
        grid_in_channels: int, 
        grid_out_channels: int, 
        hidden_dim: int = 128, 
        num_processor_layers: int = 4
    ):
        """
        Args:
            grid_in_channels: Number of features per grid node (channels * window_size + static features).
            grid_out_channels: Number of prediction features (channels * prediction_steps).
            hidden_dim: Latent dimension in the mesh.
            num_processor_layers: Number of message passing steps in the mesh.
        """
        super().__init__()
        
        # Grid node embedding
        self.grid_embed = nn.Sequential(
            nn.Linear(grid_in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Mesh node initial state (learnable parameter)
        # Assuming we pass the number of mesh nodes later, but it's easier to dynamically
        # initialize based on the graph size passed during forward, or use a dummy feature vector.
        # We will map Grid -> Mesh, so initial mesh features can just be zeros before the encoder,
        # but PyG SAGEConv needs input features. We'll use a learnable embedding.
        
        # Encoder (Grid -> Mesh)
        self.encoder = BipartiteGNN(hidden_dim, hidden_dim)
        
        # Processor (Mesh -> Mesh)
        self.processor_layers = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim) for _ in range(num_processor_layers)
        ])
        self.processor_norms = nn.ModuleList([
            LayerNorm(hidden_dim) for _ in range(num_processor_layers)
        ])
        
        # Conditioning MLPs to inject Nino 3.4 index (scalar) into the mesh
        self.condition_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder (Mesh -> Grid)
        self.decoder = BipartiteGNN(hidden_dim, hidden_dim)
        
        # Final output projection
        self.output_proj = nn.Linear(hidden_dim, grid_out_channels)
        
    def forward(
        self, 
        x_grid: torch.Tensor, 
        condition: torch.Tensor, 
        edge_grid_to_mesh: torch.Tensor, 
        edge_mesh_to_mesh: torch.Tensor, 
        edge_mesh_to_grid: torch.Tensor,
        num_mesh_nodes: int
    ) -> torch.Tensor:
        """
        Args:
            x_grid: [batch, num_grid_nodes, grid_in_channels]
            condition: [batch, 1] (Nino 3.4 index)
            edge_grid_to_mesh: [2, num_edges]
            edge_mesh_to_mesh: [2, num_edges]
            edge_mesh_to_grid: [2, num_edges]
            num_mesh_nodes: Int, number of mesh nodes.
        """
        batch_size = x_grid.size(0)
        num_grid_nodes = x_grid.size(1)
        
        # 1. Embed Grid
        # Shape: [batch * num_grid_nodes, hidden_dim]
        grid_feat = self.grid_embed(x_grid.view(-1, x_grid.size(-1)))
        
        # 2. Initialize Mesh Features
        # Using zeros as initial state before receiving grid messages
        mesh_feat = torch.zeros((batch_size * num_mesh_nodes, grid_feat.size(-1)), device=x_grid.device)
        
        # 3. Encoder: Grid -> Mesh (vectorised batch offset)
        batched_g2m = _batch_edges(edge_grid_to_mesh, num_grid_nodes, num_mesh_nodes, batch_size)
        mesh_feat = self.encoder(grid_feat, mesh_feat, batched_g2m)
        
        # 4. Processor: Mesh -> Mesh with Conditioning
        batched_m2m = _batch_edges(edge_mesh_to_mesh, num_mesh_nodes, num_mesh_nodes, batch_size)
        
        # Prepare condition [batch * num_mesh_nodes, hidden_dim]
        cond_embed = self.condition_mlp(condition) # [batch, hidden_dim]
        # Expand to mesh nodes
        cond_embed = cond_embed.unsqueeze(1).expand(-1, num_mesh_nodes, -1).reshape(-1, grid_feat.size(-1))
        
        for conv, norm in zip(self.processor_layers, self.processor_norms):
            # Inject condition by adding it
            mesh_feat = mesh_feat + cond_embed
            
            # Message pass
            msg = conv(mesh_feat, batched_m2m)
            msg = norm(msg)
            msg = torch.nn.functional.gelu(msg)
            
            # Residual connection
            mesh_feat = mesh_feat + msg
            
        # 5. Decoder: Mesh -> Grid (vectorised batch offset)
        batched_m2g = _batch_edges(edge_mesh_to_grid, num_mesh_nodes, num_grid_nodes, batch_size)
        
        # Re-initialize target grid features for decoder
        decoder_grid_feat = torch.zeros_like(grid_feat)
        out_grid_feat = self.decoder(mesh_feat, decoder_grid_feat, batched_m2g)
        
        # Add residual from input grid embed
        out_grid_feat = out_grid_feat + grid_feat
        
        # 6. Output Projection
        preds = self.output_proj(out_grid_feat) # [batch * num_grid_nodes, grid_out_channels]
        
        return preds.view(batch_size, num_grid_nodes, -1)
