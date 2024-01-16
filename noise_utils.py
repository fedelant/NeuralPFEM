"""
Utils to add noise to inputs, making simulator more stable for long rollouts. 
"""

import torch
from gns import learned_simulator

def get_random_walk_noise_for_position_and_velocity_sequence(
        position_sequence: torch.tensor,
        velocity_sequence: torch.tensor,
        noise_std_last_step):
  
  """
  Returns random-walk noise applied to the position and velocity.

  Args: 
    position_sequence: A sequence of particle positions. Shape is
      (nparticles, M, dim). Includes current + last positions.
    noise_std_last_step: Standard deviation of noise in the last step.
  """
  
  # TO DO check
  sequence_len = position_sequence.shape[1] - 1
  velocity_sequence_noise = torch.randn(
      list(velocity_sequence.shape)) * (noise_std_last_step/sequence_len**0.5)

  # Apply the random walk.
  velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

  # Adding no noise to the very first position (since that will only be used to calculate the first position change).
  # TO DO 
  position_sequence_noise = torch.cat([
      torch.zeros_like(velocity_sequence_noise[:, 0:1]),
      torch.cumsum(velocity_sequence_noise[:, 1:], dim=1)], dim=1)

  return position_sequence_noise, velocity_sequence_noise