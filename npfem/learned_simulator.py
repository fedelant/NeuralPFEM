import numpy as np
import sys
from typing import Dict

import torch
import torch.nn as nn

from npfem import graph_network
import mesh_gen


class LearnedSimulator(nn.Module):
  
  def __init__(
          self,
          nnode_in: int,
          nedge_in: int,
          latent_dim: int,
          n_message_passing_steps: int,
          nmlp_layers: int,
          mlp_hidden_dim: int,
          boundary: torch.tensor,
          normalization_stats: dict,
          dt: float,          
          h: float,
          velocity_scale_weight: float,
          boundary_clamp_limit: float,
          ar_freq: int,
          device="cuda"
  ):
    
    """
    Initializes the model.

    Args:
      nnode_in: number of node features inputs
      nedge_in: number of edge features inputs
      latent_dim: dimension of latent node/edge data
      n_message_passing_steps: number of message passing steps
      nmlp_layers: number of hidden layers in the MLP 
      boundaries: Array of 2-tuples, containing the lower and upper boundaries
        of the cuboid containing the particles TO DO
      normalization_stats: TO DO check Dictionary with statistics with keys "acceleration"
        and "velocity", containing a named tuple for each with mean and std
        fields, matching the dimensionality of the problem
      boundary_clamp_limit: a factor to enlarge connectivity radius used for computing
        normalized clipped distance in edge feature TO DO
      device: runtime device (cuda or cpu)

    """

    super(LearnedSimulator, self).__init__()
    self._boundary = boundary.to(device)
    self._normalization_stats = normalization_stats
    self._velocity_scale_weight = velocity_scale_weight
    self._h = h
    self.dt = dt

    self._ar_freq = ar_freq

    self._boundary_clamp_limit = boundary_clamp_limit

    # Initialize the EncodeProcessDecode
    self._encode_process_decode = graph_network.EncodeProcessDecode(
        nnode_in_features=nnode_in,
        nnode_out_features=2, #2D TO DO extend to 3D
        nedge_in_features=nedge_in,
        latent_dim=latent_dim,
        nmessage_passing_steps=n_message_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim)

    self._device = device

  def forward(self):

    """Forward hook runs on class instantiation"""

    pass

  def _adjust_graph_edges(
          self,
          edges: torch.tensor,
          n_particles_per_example: torch.tensor,
          n_edges_per_example: torch.tensor):
    
    """
    Adjust the particle indices in the edges according to the batch size

    Args:
      edges: edges
      n_particles_per_example: number of particles per example.
      n_edges_per_example:number of edges per example.
    """

    # Cumulative indexing adopted to avoid connections between particles of different examples
    n_cum_edges=0 
    for j in range(n_particles_per_example.shape[0]-1):
        n_cum_edges += n_edges_per_example[j]
        edges[n_cum_edges:] += n_particles_per_example[j]

    senders = edges[:,0].long()
    receivers = edges[:,1].long()

    # The flow direction when using in combination with message passing is "source_to_target" TO DO invertire
    return torch.stack([senders, receivers])
    
  def _compute_graph_connectivity(
          self,
          position: torch.tensor,
          velocity: torch.tensor,
          bounds: torch.tensor,
          free_surf: torch.tensor,
          step):
    
    """
    Generate graph edges using the PFEM mesh generator (FORTRAN bind)

    Args:
      postion: particles position with shape (nparticles, dim).
      nparticles_per_example: number of particles per example. Default is 2 examples per batch.
      radius: threshold to construct edges between all particles within the radius.
      add_self_edges: Boolean flag to include self edge (default: True)
    """

    # Definition of data structures
    # Torch tensor must be converted to numpy array
    # The number of cells and edges is not known a priori
    cells = np.zeros((position.shape[0]*3, 3), dtype = np.int32)
    edges = np.zeros((position.shape[0]*3*9, 2), dtype = np.int32)
    
    np_pos = position.cpu().detach().numpy()
    np_vel = velocity.cpu().detach().numpy()
    np_free_surf = free_surf.cpu().detach().numpy()
    np_bounds = bounds.cpu().detach().numpy()
    bound_cumsum=np_bounds.cumsum()

    # Call the PFEM mesher
    np_pos, np_vel, np_free_surf, cells, edges= mesh_gen.tassellation(np_pos, np_vel, np_free_surf, np_bounds, cells, edges, step, self._ar_freq, np_pos.shape[0])
  
    # Delete the useless rows
    cells = cells[~np.all(cells == 0, axis=1)]
    edges = edges[~np.all(edges == 0, axis=1)]
    
    # Delete repeated edges
    np_edges = np.unique(edges, axis=0)
    np_edges = np_edges.astype(np.int64)
    np_edges2 = np.array(np_edges)

    for i in range(np_edges.shape[0]):
          np_edges2[i,0]=np_edges[i,0]-bound_cumsum[np_edges[i,0]]
          np_edges2[i,1]=np_edges[i,1]-bound_cumsum[np_edges[i,1]]
    
    # TO DO verificare cosa meglio
    #senders = torch.tensor(np_edges[:,0], device=torch.device('cuda'))
    #receivers = torch.tensor(np_edges[:,1], device=torch.device('cuda'))
    #free_surf = torch.tensor(np_free_surf, device=torch.device('cuda'))
    senders = torch.from_numpy(np_edges2[:,0]).cuda(0)
    receivers = torch.from_numpy(np_edges2[:,1]).cuda(0)
    free_surf = torch.from_numpy(np_free_surf).cuda(0)
    
    # The flow direction when using in combination with message passing is "source_to_target"
    return np_pos, np_vel, torch.stack([senders, receivers]), cells, free_surf

  def _preprocessor(
          self,
          current_position: torch.tensor,
          velocity_sequence: torch.tensor,
          bounds: torch.tensor,
          edge_index: torch.tensor):
          #free_surf: torch.tensor,
    
    """
    Extracts features used for training and predictions.
    Returns a tuple of node_features (nparticles, TO DO), and edge_features (nparticles, 3).

    Args:
      position_sequence: a sequence of particle positions. Includes current + last M positions
      
      nparticles_per_example: Number of particles per example. Default is 2 examples per batch.
    """
    
    n_particles = current_position.shape[0]

    # Initialize node features
    node_features = []

    # Normalized clipped distances to lower-left and upper-right boundaries (BCs)
    boundary = self._boundary.clone().detach().requires_grad_(False).to(self._device)
    distance_to_lower_boundary = torch.abs(
        current_position - boundary[:, 0][None])
    distance_to_upper_boundary = torch.abs(
        boundary[:, 1][None] - current_position)
    distance_to_boundary = torch.cat(
        [distance_to_lower_boundary, distance_to_upper_boundary], dim=1)
    normalized_clipped_distance_to_boundary = torch.clamp(
        distance_to_boundary / self._h,
        -self._boundary_clamp_limit, self._boundary_clamp_limit)
    
    # Scaling velocity input sequence with a weight
    velocity_sequence = velocity_sequence * self._velocity_scale_weight
    
    node_features.append(velocity_sequence.view(n_particles, -1))
    node_features.append(normalized_clipped_distance_to_boundary)
    #node_features.append(free_surf.view(n_particles,1)) TO DO free surf
    
    # Initialize edge features.
    edge_features = []

    # Relative displacement and distances normalized to radius with shape (n_edges, 2)
    normalized_relative_displacements = (
        current_position[edge_index[0], :] -
        current_position[edge_index[1], :]
    )  / self._h

    # Relative distance between 2 particles with shape (n_edges, 1)
    normalized_relative_distances = torch.norm(
        normalized_relative_displacements, dim=-1, keepdim=True)

    # Edge features has a final shape of (n_edges, 3)
    edge_features.append(normalized_relative_displacements)
    edge_features.append(normalized_relative_distances)

    return (torch.cat(node_features, dim=-1),
            torch.cat(edge_features, dim=-1))

  def learned_update(
          self,
          current_position: torch.tensor,
          current_velocity: torch.tensor,
          bounds: torch.tensor,
          free_surf: torch.tensor,
          step) -> torch.tensor:
    
    """
    Predict position to evaluate the model based on position/velocity

    Args:
      current_position: Current particle positions (n_particles, dim).
      bounds: Vector indicating boundary particles. Shape: (n_particles).
      free_surf: Vector indicating free surface particles. Shape: (n_particles).

    Returns:
      predicted_position (torch.tensor): Next predicted position of particles.
      predicted_velocity (torch.tensor): Next predicted velocity of particles.
      cells (torch.tensor): New mesh/graph generated.
      free_surf (torch.tensor): Vector indicating new free surface particles.
    """
    
    # Compute the graph connectivity using the PFEM mesher
    new_pos, new_vel, edge_index, cells, free_surf = self._compute_graph_connectivity(current_position, current_velocity[:,-1], bounds, free_surf, step)
    new_pos_torch = torch.from_numpy(new_pos).cuda(0)
    current_position = new_pos_torch
    new_vel_torch = torch.from_numpy(new_vel).cuda(0)
    current_velocity[:,-1] = new_vel_torch

    # Assign features to nodes and edges
    node_features, edge_features = self._preprocessor(
            current_position[bounds==0], current_velocity[bounds==0], bounds, edge_index)
    
    predicted_normalized_velocity = self._encode_process_decode(
        node_features, edge_index, edge_features)

    predicted_velocity = (
        predicted_normalized_velocity * self._normalization_stats['std']
    ) + self._normalization_stats['mean']

    # Initialize the new tensor with zeros
    zero_tensor = torch.zeros_like(current_velocity[:,-1])

    # Indexes in the new tensor to place original tensor values
    original_tensor_indices = torch.nonzero(~bounds, as_tuple=False).squeeze()

    # Fill in the new tensor with original tensor values at appropriate positions
    zero_tensor[original_tensor_indices] = predicted_velocity
    
    #predicted_velocity = torch.where(torch.stack((bounds, bounds), dim=1), zero_tensor, predicted_velocity)
    
    predicted_position = current_position + zero_tensor * self.dt 
    return predicted_position, new_pos_torch, new_vel_torch, zero_tensor, cells, free_surf

  def predict_velocity(
          self,
          next_velocity: torch.tensor,
          current_position: torch.tensor,
          velocity_sequence: torch.tensor,
          edges:torch.tensor,
          bounds:torch.tensor,
          n_particles_per_example: torch.tensor,
          n_edges_per_example: torch.tensor):
          #free_surf:torch.tensor,
    
    """
    Predict velocity to train the model based on

    Args:
      next_velocities: Tensor of shape (nparticles_in_batch, dim) with the
        velocities the model should output given the inputs
      position_sequence_noise: Tensor of the same shape as `position_sequence`
        with the noise to apply to each particle.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, , dim). Includes current + last  positions.
      velcity_sequence_noise: Tensor of the same shape as `_sequence`
        with the noise to apply to each 
      velocity_sequence: A sequence of particle positions. Shape is
        (nparticles, , dim). Includes current + last  positions.
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.

    Returns:
      Tensors of shape (nparticles_in_batch, dim) with the predicted and target
        normalized velocities.

    """

    # Adjust the edge indices depending on the batch dimension
    edge_index = self._adjust_graph_edges(edges, n_particles_per_example, n_edges_per_example)
    #print(edge_index.shape)

    # Assign features to nodes and edges
    node_features, edge_features = self._preprocessor(current_position, velocity_sequence, bounds, edge_index)

    # Perform the forward pass with the GNN model
    predicted_normalized_velocity = self._encode_process_decode(node_features, edge_index, edge_features)
    
    # Calculate the target normalized velocity
    target_normalized_velocity = (next_velocity - self._normalization_stats['mean']) / self._normalization_stats['std']
    
    return predicted_normalized_velocity, target_normalized_velocity

  def save(
          self,
          path: str = 'model.pt'):
    """Save model state

    Args:
      path: Model path
    """
    torch.save(self.state_dict(), path)

  def load(
          self,
          path: str):
    """Load model state from file

    Args:
      path: Model path
    """
    self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))