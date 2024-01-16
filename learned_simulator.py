"""
Class implementing the GNS that learns the particles state update 
"""

import torch
import torch.nn as nn
import numpy as np
from gns import graph_network
from torch_geometric.nn import radius_graph
from typing import Dict

class LearnedSimulator(nn.Module):
  
  def __init__(
          self,
          problem_dim: int,
          nnode_in: int,
          nedge_in: int,
          latent_dim: int,
          nmessage_passing_steps: int,
          nmlp_layers: int,
          mlp_hidden_dim: int,
          connectivity_radius: float,
          boundaries: np.ndarray,
          normalization_stats: dict,
          dt: float,
          boundary_clamp_limit: float = 1.0, #TO DO: 
          device="cpu"
  ):
    
    """
    Initializes the model.

    Args:
      problem_dim: dimensionality of the problem
      nnode_in: number of node features inputs
      nedge_in: number of edge features inputs
      latent_dim: dimension of latent node/edge data
      nmessage_passing_steps: number of message passing steps
      nmlp_layers: number of hidden layers in the MLP 
      connectivity_radius: scalar with the radius of connectivity
      boundaries: Array of 2-tuples, containing the lower and upper boundaries
        of the cuboid containing the particles TO DO
      normalization_stats: TO DO check Dictionary with statistics with keys "acceleration"
        and "velocity", containing a named tuple for each with mean and std
        fields, matching the dimensionality of the problem
      dt: time step of the simulation data
      boundary_clamp_limit: a factor to enlarge connectivity radius used for computing
        normalized clipped distance in edge feature TO DO
      device: runtime device (cuda or cpu)
    """

    super(LearnedSimulator, self).__init__()
    self._boundaries = boundaries
    self._connectivity_radius = connectivity_radius
    self._normalization_stats = normalization_stats
    self._boundary_clamp_limit = boundary_clamp_limit
    self.dt = dt

    # Initialize the EncodeProcessDecode.
    self._encode_process_decode = graph_network.EncodeProcessDecode(
        nnode_in_features=nnode_in,
        nnode_out_features=problem_dim,
        nedge_in_features=nedge_in,
        latent_dim=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim)

    self._device = device

  def forward(self):

    """
    Forward hook runs on class instantiation
    """

    pass

  def _compute_graph_connectivity(
          self,
          position: torch.tensor,
          nparticles_per_example: torch.tensor,
          radius: float,
          add_self_edges: bool = True):
    
    """
    Generate graph edges between all the particles within a threshold radius.

    Args:
      postion: particles position with shape (nparticles, dim).
      nparticles_per_example: number of particles per example. Default is 2 examples per batch.
      radius: threshold to construct edges between all particles within the radius.
      add_self_edges: Boolean flag to include self edge (default: True).
    """

    # Specify examples id for particles.
    batch_ids = torch.cat(
        [torch.LongTensor([i for _ in range(n)])
         for i, n in enumerate(nparticles_per_example)]).to(self._device)

    # Radius_graph generate a graph connectig nodes if the distance is < r.
    # The result is a torch tensor list of source and target nodes with shape (2, nedges)
    edge_index = radius_graph(
        position, r=radius, batch=batch_ids, loop=add_self_edges, max_num_neighbors=128)

    # The flow direction when using in combination with message passing is "source_to_target"
    receivers = edge_index[0, :]
    senders = edge_index[1, :]
    
    # TO DO check
    return receivers, senders

  def _preprocessor(
          self,
          position_sequence: torch.tensor,
          velocity_sequence: torch.tensor,
          nparticles_per_example: torch.tensor):
    
    """
    Extracts features from the position/velocity sequence used for training and predictions.
    Returns a tuple of node_features (nparticles, TO DO), edge_index (nparticles, nparticles)
    and edge_features (nparticles, 3).

    Args:
      position_sequence: a sequence of particle positions. Includes current + last M positions
      position_sequence: a sequence of particle velocities. Includes current + last M velocities
        Shape of thes 2 is (nparticles, M+1, dim)
      nparticles_per_example: Number of particles per example. Default is 2 examples per batch.
    """

    nparticles = position_sequence.shape[0] # TO DO necessary?
    most_recent_position = position_sequence[:, -1] 

    # Create graph.
    senders, receivers = self._compute_graph_connectivity(
        most_recent_position, nparticles_per_example, self._connectivity_radius)
    
    # Initialize node features.
    node_features = []

    # TO DO: trovare migliore node feature
    positon_diff_sequence = time_diff(position_sequence)
    normalized_velocity_sequence = (
      velocity_sequence - self._normalization_stats['mean']) / self._normalization_stats['std']
    node_features.append(position_sequence.view(nparticles, -1))

    # Normalized clipped distances to lower and upper boundaries. Boundaries are an array of 
    # shape [num_dimensions, 2], where the second axis, provides the lower/upper boundaries.
    # TO DO: capire, normalization
    boundaries = torch.tensor(
        self._boundaries, requires_grad=False).float().to(self._device)
    distance_to_lower_boundary = (
        most_recent_position - boundaries[:, 0][None])
    distance_to_upper_boundary = (
        boundaries[:, 1][None] - most_recent_position)
    distance_to_boundaries = torch.cat(
        [distance_to_lower_boundary, distance_to_upper_boundary], dim=1)
    normalized_clipped_distance_to_boundaries = torch.clamp(
        distance_to_boundaries / self._connectivity_radius,
        -self._boundary_clamp_limit, self._boundary_clamp_limit)
    node_features.append(normalized_clipped_distance_to_boundaries)

    # Collect edge features.
    edge_features = []

    # Add relative displacement normalized with connectivity radius  as an edge feature with shape (nparticles, ndim)/(nedges, 2).
    # TO DO: check shape,provare con relative velocities e senza norm
    normalized_relative_displacements = (
        most_recent_position[senders, :] -
        most_recent_position[receivers, :]
        ) / self._connectivity_radius
    edge_features.append(normalized_relative_displacements)

    # Add relative distance between 2 particles as an edge feature with shape (nparticles, 1).
    normalized_relative_distances = torch.norm(normalized_relative_displacements, dim=-1, keepdim=True)
    edge_features.append(normalized_relative_distances)

    return (torch.cat(node_features, dim=-1),
            torch.stack([senders, receivers]),
            torch.cat(edge_features, dim=-1))

  def learned_update(
          self,
          current_position: torch.tensor,
          current_velocity: torch.tensor,
          nparticles: torch.tensor) -> torch.tensor:
    
    """
    Predict position to evaluate the model based on position/velocity.

    Args:
      current_position: current particle positions (nparticles, dim).
      current_velocity: current particle velocities (nparticles, dim).
      nparticles: number of particles. 

    Returns:
      predicted_position (torch.tensor): next predicted position of particles.
      predicted_velocity (torch.tensor): next predicted velocity of particles.
    """

    node_features, edge_index, edge_features = self._preprocessor(
            current_position, current_velocity, nparticles)
    predicted_normalized_velocity = self._encode_process_decode(
        node_features, edge_index, edge_features)
    predicted_velocity = (
        predicted_normalized_velocity * self._normalization_stats['std']
    ) + self._normalization_stats['mean']

    predicted_position = current_position[:, -1] + predicted_velocity * self.dt
    return predicted_position, predicted_velocity

  def predict_velocity(
          self,
          next_velocity: torch.tensor,
          position_sequence_noise: torch.tensor,
          position_sequence: torch.tensor,
          velocity_sequence_noise: torch.tensor,
          velocity_sequence: torch.tensor,
          particle_move_sequence:torch.tensor,
          nparticles_per_example: torch.tensor):
    
    """
    Predict next velocity to train the model using a sequence of M previous positions/velocities.

    Args:
      next_velocity: Tensor of shape (nparticles_in_batch, dim) with the
        velocity the model should output given the inputs.
      position_sequence_noise: Tensor of the same shape as `position_sequence`
        with the noise to apply to each particle position.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, , dim). Includes current + last  positions.
      velcity_sequence_noise: Tensor of the same shape as `_sequence`
        with the noise to apply to each particle velocity.
      velocity_sequence: A sequence of particle positions. Shape is
        (nparticles, , dim). Includes current + last  positions.
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.

    Returns:
      Tensors of shape (nparticles_in_batch, dim) with the predicted and target
        normalized velocities.
    """

    # Add noise and filter the moved particles in the input velocity/position sequence.
    noisy_position_sequence = position_sequence + position_sequence_noise
    rows_to_keep = ~torch.any(particle_move_sequence, dim=1)
    filtered_tensor = noisy_position_sequence[rows_to_keep]
    noisy_velocity_sequence = velocity_sequence + velocity_sequence_noise
    filtered_tensor_vel = noisy_velocity_sequence[rows_to_keep]

    # Compute nparticles in batch examples after filtering.
    cumsum = torch.cumsum(rows_to_keep, dim=0)
    nparticles_per_example[1] = cumsum[nparticles_per_example[0]+ nparticles_per_example[1] -1 ]
    nparticles_per_example[0] = cumsum[nparticles_per_example[0] - 1]
    nparticles_per_example[1] = nparticles_per_example[1] - nparticles_per_example[0]

    # Perform the forward pass with the noisy position/velocity sequence.
    node_features, edge_index, edge_features = self._preprocessor(
            filtered_tensor, filtered_tensor_vel, nparticles_per_example)
    predicted_normalized_velocity = self._encode_process_decode(
        node_features, edge_index, edge_features)

    # Calculate the target velocity.
    # TO DO: check se meglio noisy o no
    next_velocity_adjusted = next_velocity + velocity_sequence_noise[:, -1]
    next_velocity_adjusted = next_velocity_adjusted[rows_to_keep]
    target_normalized_velocity = (next_velocity_adjusted - self._normalization_stats['mean']) / self._normalization_stats['std']

    return predicted_normalized_velocity, target_normalized_velocity

  def save(
          self,
          path: str = 'model.pt'):
    
    """
    Save model state.

    Args:
      path: Model path.
    """
    
    torch.save(self.state_dict(), path)

  def load(
          self,
          path: str):
    
    """
    Load model state from file

    Args:
      path: Model path
    """

    self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

def time_diff(
        position_sequence: torch.tensor) -> torch.tensor:
   
   """
   Finite difference in time between two input position sequence.

   Args:
     position_sequence: Input position sequence & shape(nparticles, M, dim).

   Returns:
     torch.tensor: Position time difference sequence.
   """
   
   return position_sequence[:, 1:] - position_sequence[:, :-1]