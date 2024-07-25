import torch

def get_random_walk_noise_for_position_and_velocity_sequence(
        position_sequence: torch.tensor,
        velocity_sequence: torch.tensor,
        noise_std_weight):
  
  """
  Returns random-walk noise in the velocity applied to the velocity.

  Args: 
    TO DO 
    position_sequence: A sequence of particle positions. Shape is
      (nparticles, 6, dim). Includes current + last 5 positions.
    position_sequence: A sequence of particle positions. Shape is
      (nparticles, 6, dim). Includes current + last 5 positions.
    noise_std_last_weight: .

  """

  mean_abs_vel = float(torch.mean(torch.abs(velocity_sequence)))
  velocity_sequence_noise = torch.randn(list(velocity_sequence[:, 1:-1, :].shape)) * (mean_abs_vel/noise_std_weight)
  velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)
  
  velocity_sequence_noise = torch.cat([
      torch.zeros_like(velocity_sequence_noise[:, 0:2]),
      velocity_sequence_noise], dim=1)
  # adding no noise to the very first position 
  # (since that will only be used to calculate the first position change).
  position_sequence_noise = torch.cat([
      torch.zeros_like(velocity_sequence_noise[:, 0:1]),
      torch.cumsum(velocity_sequence_noise[:, 1:], dim=1)], dim=1)

  return position_sequence_noise, velocity_sequence_noise