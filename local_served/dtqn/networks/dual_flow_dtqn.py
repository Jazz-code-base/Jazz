import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Tuple

from dtqn.networks.representations import (
    ObservationEmbeddingRepresentation,
    ActionEmbeddingRepresentation,
)
from dtqn.networks.position_encodings import PosEnum, PositionEncoding
from dtqn.networks.gates import GRUGate, ResGate
from dtqn.networks.transformer import TransformerLayer, TransformerIdentityLayer
from utils import torch_utils


class DualFlowDTQN(nn.Module):
    """Dual-flow version of DTQN network, specifically designed to handle input data from two subflows
    
    Args:
        obs_dim:            Length of the state vector (total length, including both subflows)
        num_actions:        Number of possible actions
        embed_per_obs_dim:  Embedding length for each observation dimension in discrete observation spaces
        action_dim:         Number of features for actions
        inner_embed_size:   Network dimensionality
        num_heads:          Number of heads in multi-head attention
        num_transformer_layers:         Number of transformer blocks
        history_len:        Maximum observation history length
        dropout:            Dropout ratio
        gate:               Layer type used after attention and feedforward submodules ('res' or 'gru')
        identity:           Whether to use identity mapping reordering
        pos:                Position encoding type
        discrete:           Whether the environment has discrete observations
        vocab_sizes:        For discrete environments, represents the number of observations
        bag_size:           Size of the bag mechanism
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        embed_per_obs_dim: int,
        action_embed_dim: int,
        inner_embed_size: int,
        num_heads: int,
        num_transformer_layers: int,
        history_len: int,
        dropout: float = 0.0,
        gate: str = "res",
        identity: bool = False,
        pos: Union[str, int] = "learned",
        discrete: bool = False,
        vocab_sizes: Optional[Union[np.ndarray, int]] = None,
        bag_size: int = 0,
        subflow_hidden_dim: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.discrete = discrete
        
        # Set each subflow dimension to half of the total dimension
        self.subflow_dim = obs_dim // 2
        
        # Subflow 1 embedding network
        self.subflow1_fc1 = nn.Linear(self.subflow_dim, subflow_hidden_dim)
        self.subflow1_fc2 = nn.Linear(subflow_hidden_dim, subflow_hidden_dim)
        
        # Subflow 2 embedding network
        self.subflow2_fc1 = nn.Linear(self.subflow_dim, subflow_hidden_dim)
        self.subflow2_fc2 = nn.Linear(subflow_hidden_dim, subflow_hidden_dim)
        
        # Mapping after merging subflow features
        self.subflow_merge = nn.Linear(subflow_hidden_dim * 2, inner_embed_size - action_embed_dim)
        
        # Input embedding: allocate space for action embedding
        if action_embed_dim > 0:
            self.action_embedding = ActionEmbeddingRepresentation(
                num_actions=num_actions, action_dim=action_embed_dim
            )
        else:
            self.action_embedding = None

        # Position encoding
        pos_function_map = {
            PosEnum.LEARNED: PositionEncoding.make_learned_position_encoding,
            PosEnum.SIN: PositionEncoding.make_sinusoidal_position_encoding,
            PosEnum.NONE: PositionEncoding.make_empty_position_encoding,
        }
        self.position_embedding = pos_function_map[PosEnum(pos)](
            context_len=history_len, embed_dim=inner_embed_size
        )

        self.dropout = nn.Dropout(dropout)

        # Create gating mechanism
        if gate == "gru":
            attn_gate = GRUGate(embed_size=inner_embed_size)
            mlp_gate = GRUGate(embed_size=inner_embed_size)
        elif gate == "res":
            attn_gate = ResGate()
            mlp_gate = ResGate()
        else:
            raise ValueError("Gate must be one of `gru`, `res`")

        # Create Transformer blocks
        if identity:
            transformer_block = TransformerIdentityLayer
        else:
            transformer_block = TransformerLayer
        self.transformer_layers = nn.Sequential(
            *[
                transformer_block(
                    num_heads,
                    inner_embed_size,
                    history_len,
                    dropout,
                    attn_gate,
                    mlp_gate,
                )
                for _ in range(num_transformer_layers)
            ]
        )

        # Bag mechanism setup
        self.bag_size = bag_size
        self.bag_attn_weights = None
        if bag_size > 0:
            self.bag_attention = nn.MultiheadAttention(
                inner_embed_size,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.ffn = nn.Sequential(
                nn.Linear(inner_embed_size * 2, inner_embed_size),
                nn.ReLU(),
                nn.Linear(inner_embed_size, inner_embed_size // 2),
                nn.ReLU(),
                nn.Linear(inner_embed_size // 2, num_actions),
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(inner_embed_size, inner_embed_size),
                nn.ReLU(),
                nn.Linear(inner_embed_size, inner_embed_size // 2),
                nn.ReLU(),
                nn.Linear(inner_embed_size // 2, num_actions),
            )

        self.history_len = history_len
        self.apply(torch_utils.init_weights)

    def _process_subflows(self, obss: torch.Tensor) -> torch.Tensor:
        """Process dual subflow input data
        
        Args:
            obss: Observation tensor with shape [batch x seq_len x obs_dim]
            
        Returns:
            Processed feature tensor with shape [batch x seq_len x embed_size]
        """
        batch_size, seq_len = obss.shape[0], obss.shape[1]
        
        # Split state into two subflows
        # Split obs_dim dimension in half: [batch x seq_len x obs_dim] -> [batch x seq_len x (obs_dim/2)]
        subflow1_x = obss[:, :, :self.subflow_dim]
        subflow2_x = obss[:, :, self.subflow_dim:]
        
        # Process shape for batch processing
        # [batch x seq_len x (obs_dim/2)] -> [batch*seq_len x (obs_dim/2)]
        subflow1_x = subflow1_x.reshape(-1, self.subflow_dim)
        subflow2_x = subflow2_x.reshape(-1, self.subflow_dim)
        
        # Subflow 1 processing
        subflow1_x = torch.relu(self.subflow1_fc1(subflow1_x))
        subflow1_x = torch.relu(self.subflow1_fc2(subflow1_x))
        
        # Subflow 2 processing
        subflow2_x = torch.relu(self.subflow2_fc1(subflow2_x))
        subflow2_x = torch.relu(self.subflow2_fc2(subflow2_x))
        
        # Combine features [batch*seq_len x subflow_hidden_dim*2]
        combined = torch.cat([subflow1_x, subflow2_x], dim=1)
        
        # Map to required embedding size [batch*seq_len x (inner_embed_size-action_dim)]
        merged = self.subflow_merge(combined)
        
        # Restore original shape [batch x seq_len x (inner_embed_size-action_dim)]
        return merged.reshape(batch_size, seq_len, -1)

    def forward(
        self,
        obss: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        bag_obss: Optional[torch.Tensor] = None,
        bag_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass
        
        Args:
            obss:       Observation tensor with shape [batch x seq_len x obs_dim]
            actions:    Action tensor with shape [batch x seq_len x 1]
            bag_obss:   Bag observation tensor with shape [batch x bag_size x obs_dim]
            bag_actions: Bag action tensor with shape [batch x bag_size x 1]
            
        Returns:
            Q-value tensor with shape [batch x seq_len x num_actions]
        """
        history_len = obss.size(1)
        assert (
            history_len <= self.history_len
        ), "Cannot forward, history is longer than expected."

        # Check observation dimensions
        obs_dim = obss.size()[2:] if len(obss.size()) > 3 else obss.size(2)
        assert (
            obs_dim == self.obs_dim
        ), f"Obs dim is incorrect. Expected {self.obs_dim} got {obs_dim}"

        # Process subflows
        token_embeddings = self._process_subflows(obss)

        # If action embedding exists, add action information
        if self.action_embedding is not None:
            # [batch x seq_len x 1] -> [batch x seq_len x action_embed]
            action_embed = self.action_embedding(actions)

            if history_len > 1:
                action_embed = torch.roll(action_embed, 1, 1)
                # First observation in sequence has no previous action, so zero out features
                action_embed[:, 0, :] = 0.0
            token_embeddings = torch.concat([action_embed, token_embeddings], dim=-1)

        # Process through Transformer layers
        # [batch x seq_len x model_embed] -> [batch x seq_len x model_embed]
        working_memory = self.transformer_layers(
            self.dropout(
                token_embeddings + self.position_embedding()[:, :history_len, :]
            )
        )

        # Process Bag mechanism
        if self.bag_size > 0 and bag_obss is not None:
            # Process observations and actions in the bag
            bag_token_embeddings = self._process_subflows(bag_obss)
            
            if self.action_embedding is not None:
                bag_embeddings = torch.concat(
                    [self.action_embedding(bag_actions), bag_token_embeddings],
                    dim=-1,
                )
            else:
                bag_embeddings = bag_token_embeddings
                
            # [batch x seq_len x model_embed] x [batch x bag_size x model_embed] -> [batch x seq_len x model_embed]
            persistent_memory, self.attn_weights = self.bag_attention(
                working_memory, bag_embeddings, bag_embeddings
            )
            output = self.ffn(torch.concat([working_memory, persistent_memory], dim=-1))
        else:
            output = self.ffn(working_memory)

        return output[:, -history_len:, :] 