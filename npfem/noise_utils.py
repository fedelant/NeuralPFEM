import torch

def get_random_walk_noise_for_position_and_velocity_sequence(
        position: torch.tensor,
        velocity_sequence: torch.tensor,
        noise_std_weight,
        dt):
  
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
  velocity_sequence_noise = torch.randn(list(velocity_sequence[:, 1:, :].shape)) * (mean_abs_vel/noise_std_weight)
  velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)
  velocity_sequence_noise = torch.cat([
      torch.zeros(velocity_sequence_noise.shape[0],1,velocity_sequence_noise.shape[2]),
      velocity_sequence_noise], dim=1)
  position_sequence_noise = velocity_sequence_noise[:,-1,:] * dt

  return position_sequence_noise, velocity_sequence_noise