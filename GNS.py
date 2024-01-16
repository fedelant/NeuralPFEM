"""
Code to launch GNS for training and predictions
Based on https://doi.org/10.48550/arXiv.2002.09405 (https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate) 
and https://doi.org/10.1016/j.compgeo.2023.106015 (https://github.com/geoelements/gns/)
"""

import collections
import json
import os
import pickle
import glob
import re
import sys

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from absl import flags
from absl import app

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from gns import data_loader
from gns import distribute

# Code modality definition
flags.DEFINE_enum(
    'mode', 'train', ['train', 'valid', 'test'],
    help='Train model, validation or test')

# Input/output folders/files
flags.DEFINE_string('data_path', None, help='The dataset directory')
flags.DEFINE_string('model_path', 'models/', help=('The path for saving checkpoints of the model'))
flags.DEFINE_string('output_path', 'outputs/', help='The path for saving outputs')
flags.DEFINE_string('output_filename', 'test', help='Base name for saving the output')
flags.DEFINE_string('model_file', None, help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file'))
flags.DEFINE_string('train_state_file', 'train_state.pt', help=('Train state filename (.pt) to resume from. Can also use "latest" to default to newest file'))

# Steps handling
flags.DEFINE_integer('ntraining_steps', int(2E7), help='Number of training steps.')
flags.DEFINE_integer('nsave_steps', int(5000), help='Nsteps frequency to save the model')

# Hyperparameters
flags.DEFINE_integer('batch_size', 2, help='The batch size')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise')
flags.DEFINE_float('lr_init', 1e-4, help='Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', int(5e6), help='Learning rate decay steps.')
flags.DEFINE_float('connectivity_radius', 0.04, help='Connectivity radius used to build the graph')
flags.DEFINE_integer('input_sequence_length', 6, help='Sequence length of previous velocities/positions used to predict the update.') # TO DO 
INPUT_SEQUENCE_LENGTH = 5

# CUDA device 
flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")

FLAGS = flags.FLAGS

# TO DO ?
#Stats = collections.namedtuple('Stats', ['mean', 'std'])

def predict_example(
        simulator: learned_simulator.LearnedSimulator,
        position: torch.tensor,
        velocity: torch.tensor,
        n_particles: torch.tensor,
        nsteps: int):

  """
  Generate a trajectory by applying the learned model in sequence for an example in the valid/test dataset.

  Args:
    simulator: Learned simulator.
    position: Positions of particles. Shape: (timesteps, n_particles, ndims).
    velocity: Velocities of particles. Shape: (timesteps, n_particles, ndims).
    n_particles: Number of particles in the example.
    nsteps: Number of steps.
  """
  
  # Initial configuration.
  initial_position = position[:, :INPUT_SEQUENCE_LENGTH]
  initial_velocity = velocity[:, :INPUT_SEQUENCE_LENGTH]

  # Trajectory data from the simulation that I want to predict.
  ground_truth_position = position[:, INPUT_SEQUENCE_LENGTH:]
  ground_truth_velocity = velocity[:, INPUT_SEQUENCE_LENGTH:]

  # Initialize position, velocities and trajectory vectors.
  current_position = initial_position
  current_velocity = initial_velocity
  predicted_position = []
  predicted_velocity = []

  # Cycle over the number of time steps that I want to predict.
  for step in tqdm(range(nsteps), total=nsteps):

    # Get next position and velocity, each with shape (nnodes, dim), with learned simulator.
    next_position, next_velocity = simulator.learned_update(
        current_position,
        current_velocity,
        nparticles=[n_particles]
    )

    # Add the predicted next position and velocity to the trajectory vectors.
    predicted_position.append(next_position)
    predicted_velocity.append(next_velocity)

    # Shift current_position/velocity, removing the oldest position in the sequence and appending the next position at the end.
    current_position = torch.cat(
        [current_position[:, 1:], next_position[:, None, :]], dim=1)
    current_velocity = torch.cat(
        [current_velocity[:, 1:], next_velocity[:, None, :]], dim=1)

  # Predicted position and velocity with shape (time_steps, nnodes, dim).
  predicted_position = torch.stack(predicted_position)
  predicted_velocity = torch.stack(predicted_velocity)

  # Computing the error between the simulated trajectory and the predicted one.
  # TO DO: find the best way to evaluate test error
  ground_truth_position = ground_truth_position.permute(1, 0, 2)
  ground_truth_velocity = ground_truth_velocity.permute(1, 0, 2)
  example_error = (predicted_position - ground_truth_position) ** 2

  # Output data structure
  output_dict = {
      'initial_position': initial_position.permute(1, 0, 2).cpu().numpy(),
      'initial_velocity': initial_velocity.permute(1, 0, 2).cpu().numpy(),
      'predicted_position': predicted_position.cpu().numpy(),
      'predicted_velocity': predicted_position.cpu().numpy(),
      'ground_truth_position': ground_truth_position.cpu().numpy(),
      'ground_truth_velocity': ground_truth_velocity.cpu().numpy()
  }

  return output_dict, example_error


def predict(device: str):

  """Predict and evaluate trajectories for valid/test datasets.

  Args:
    simulator: Trained simulator.

  """

  # Read metadata.
  metadata = reading_utils.read_metadata(FLAGS.data_path, "rollout") # TO DO

  # Get the valid or test dataset
  split = 'test' if FLAGS.mode == 'test' else 'valid'
  ds = data_loader.get_data_loader_by_trajectories(path=f"{FLAGS.data_path}{split}.npz")

  # Define output path.
  if not os.path.exists(FLAGS.output_path):
    os.makedirs(FLAGS.output_path)

  # Get learned simulator.
  simulator = _get_simulator(metadata, FLAGS.connectivity_radius, FLAGS.noise_std, device)
  
  if os.path.exists(FLAGS.model_path + FLAGS.model_file):
    simulator.load(FLAGS.model_path + FLAGS.model_file)
  else:
    raise Exception(f"Model does not exist at {FLAGS.model_path + FLAGS.model_file}")

  simulator.to(device)
  simulator.eval()

  # Loop over the examples in the dataset.
  error = []
  with torch.no_grad(): # TO DO: understand
    for example_i, features in enumerate(ds):
      
      nsteps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH

      # Get data from the dataset.
      position = features[0].to(device)
      velocity = features[1].to(device)
      n_particles = torch.tensor([int(features[3])], dtype=torch.int32).to(device)

      # Predict example rollout and evaluate example error loss.
      example_rollout, example_error = predict_example(simulator,
                                                       position,
                                                       velocity,
                                                       n_particles,
                                                       nsteps)
      example_rollout['metadata'] = metadata # TO DO

      # Print and save example error
      print("Predicting example {} error: {}".format(example_i, example_error.mean()))
      error.append(torch.flatten(example_error))

      # Save predicted trajectory for test phase 
      if FLAGS.mode == 'test':
        example_rollout['metadata'] = metadata # TO DO
        example_rollout['loss'] = example_error
        filename = f'{FLAGS.output_filename}_ex{example_i}.pkl'
        filename = os.path.join(FLAGS.output_path, filename)
        with open(filename, 'wb') as f:
          pickle.dump(example_rollout, f)

  print("Mean error prediction dataset: {}".format(torch.mean(torch.cat(error))))
  
  # Uncomment the following lines when, in the validation process, you are interested
  # in running this code for every model generated during training keeping track of the 
  # error history to find the best one and to evaluate the training process
  '''
  if FLAGS.mode == 'valid':
    with open("loss.txt", "a+") as file:
      file.seek(0)
      file.write(str(torch.mean(error).detach().cpu().numpy()))
      file.write("\n")
  '''

# TO DO: understand
def optimizer_to(optim, device):
  for param in optim.state.values():
    # Not sure there are any global tensors in the state dict
    if isinstance(param, torch.Tensor):
      param.data = param.data.to(device)
      if param._grad is not None:
        param._grad.data = param._grad.data.to(device)
    elif isinstance(param, dict):
      for subparam in param.values():
        if isinstance(subparam, torch.Tensor):
          subparam.data = subparam.data.to(device)
          if subparam._grad is not None:
            subparam._grad.data = subparam._grad.data.to(device)

def train(rank, flags, world_size, device):
  
  """
  Train the model.

  Args:
    rank: local rank
    world_size: total number of ranks
    device: torch device type
  """

  # Initialize CUDA device.
  if device == torch.device("cuda"):
    distribute.setup(rank, world_size, device)
    device_id = rank
  else:
    device_id = device
  
  print(f"rank = {rank}, cuda = {torch.cuda.is_available()}")

  # Read metadata.
  metadata = reading_utils.read_metadata(flags["data_path"], "train")

  # Get simulator and optimizer.
  if device == torch.device("cuda"):
    serial_simulator = _get_simulator(metadata, flags["connectivity_radius"], flags["noise_std"], rank)
    simulator = DDP(serial_simulator.to(rank), device_ids=[rank], output_device=rank)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=flags["lr_init"]*world_size)
  else:
    simulator = _get_simulator(metadata, flags["noise_std"], device)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=flags["lr_init"] * world_size)
  
  # Initialize the step index.
  step = 0

  # If model_file flag is provided, check if it is possible to continue the training.
  if flags["model_file"] is not None:

    # If model_file and train_state_file are set as "latest" search for the last model generated.
    if flags["model_file"] == "latest" and flags["train_state_file"] == "latest":
      # Find the latest model, assumes model and train_state files are in step.
      fnames = glob.glob(f'{flags["model_path"]}*model*pt')
      max_model_number = 0
      expr = re.compile(".*model-(\d+).pt")
      for fname in fnames:
        model_num = int(expr.search(fname).groups()[0])
        if model_num > max_model_number:
          max_model_number = model_num
      # Reset names to point to the latest.
      flags["model_file"] = f"model-{max_model_number}.pt"
      flags["train_state_file"] = f"train_state-{max_model_number}.pt"

    # If model_file and train_state_file exist in the indicated path, the train continue from that step.
    if os.path.exists(flags["model_path"] + flags["model_file"]) and os.path.exists(flags["model_path"] + flags["train_state_file"]):
      
      # Load model.
      if device == torch.device("cuda"):
        simulator.module.load(flags["model_path"] + flags["model_file"])
      else:
        simulator.load(flags["model_path"] + flags["model_file"])

      # Load train state.
      train_state = torch.load(flags["model_path"] + flags["train_state_file"])

      # Set optimizer state.
      optimizer = torch.optim.Adam(
        simulator.module.parameters() if device == torch.device("cuda") else simulator.parameters())
      optimizer.load_state_dict(train_state["optimizer_state"])
      optimizer_to(optimizer, device_id)

      # Update the steo index.
      step = train_state["global_train_state"].pop("step")

    # If the model or the train_state are not found in the provided path, raise an error.
    else:
      msg = f'Specified model_file {flags["model_path"] + flags["model_file"]} and train_state_file {flags["model_path"] + flags["train_state_file"]} not found.'
      raise FileNotFoundError(msg)

  # TO DO: understand
  simulator.train()
  simulator.to(device_id)

  # Get train dataset.
  if device == torch.device("cuda"):
    dl = distribute.get_data_distributed_dataloader_by_samples(path=f'{flags["data_path"]}train.npz',
                                                               input_length_sequence=INPUT_SEQUENCE_LENGTH,
                                                               batch_size=flags["batch_size"])
  else:
    dl = data_loader.get_data_loader_by_samples(path=f'{flags["data_path"]}train.npz',
                                                input_length_sequence=INPUT_SEQUENCE_LENGTH,
                                                batch_size=flags["batch_size"])

  # Loop over training steps.
  not_reached_nsteps = True
  try:
    while not_reached_nsteps:
      if device == torch.device("cuda"):
        torch.distributed.barrier()
      else:
        pass
      for example in dl:  # example = ((position, velocity, n_particles_per_example), labels).
        position = example[0][0].to(device_id)
        velocity = example[0][1].to(device_id)
        particle_move = example[0][2].to(device_id)
        n_particles_per_example = example[0][3].to(device_id)
        labels = example[1].to(device_id)
        n_particles_per_example.to(device_id)
        labels.to(device_id)

        # Sample the noise to add to the inputs to the model during training.
        # TO DO: capire errore migliore, valore, applicato a position o velocity...
        sampled_pos_noise, sampled_vel_noise = noise_utils.get_random_walk_noise_for_position_and_velocity_sequence(
          position, velocity, noise_std_last_step=flags["noise_std"])
        sampled_pos_noise = sampled_pos_noise.to(device_id)
        sampled_vel_noise = sampled_vel_noise.to(device_id)

        # Get the velocity predicted with GNN and the target velocity from data.
        if device == torch.device("cuda"):
          pred_vel, target_vel = simulator.module.predict_velocity(
            next_velocity=labels.to(rank),
            position_sequence_noise=sampled_pos_noise.to(rank),
            position_sequence=position.to(rank),
            velocity_sequence_noise=sampled_vel_noise.to(rank),
            velocity_sequence=velocity.to(rank),
            particle_move_sequence=particle_move.to(rank),
            nparticles_per_example=n_particles_per_example.to(rank)
          )
        else:
          pred_vel, target_vel = simulator.predict_velocity(
            next_velocity=labels.to(device),
            position_sequence_noise=sampled_pos_noise.to(device),
            position_sequence=position.to(device),
            velocity_sequence_noise=sampled_vel_noise.to(device),
            velocity_sequence=velocity.to(device),
            particle_move_sequence=particle_move.to(device),
            nparticles_per_example=n_particles_per_example.to(device)
          )

        # Calculate the loss (the mean of the velocity error over all the particles, exluding the recently moved ones).
        loss = (pred_vel - target_vel) ** 2
        loss = loss.sum(dim=-1)
        rows_to_keep = ~torch.any(particle_move, dim=1)
        loss = loss.sum() / rows_to_keep.sum()
        
        # Computes the gradient of loss.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update learning rate.
        lr_new = flags["lr_init"] * (flags["lr_decay"] ** (step/flags["lr_decay_steps"])) * world_size
        for param in optimizer.param_groups:
          param['lr'] = lr_new

        if rank == 0 or device == torch.device("cpu"):
          print(f'Training step: {step}/{flags["ntraining_steps"]}. Loss: {loss}.')
          # Save model state
          if step % flags["nsave_steps"] == 0:
            if device == torch.device("cpu"):
              simulator.save(flags["model_path"] + 'model-'+str(step)+'.pt')
            else:
              simulator.module.save(flags["model_path"] + 'model-'+str(step)+'.pt')
            train_state = dict(optimizer_state=optimizer.state_dict(),
                               global_train_state={"step": step},
                               loss=loss.item())
            torch.save(train_state, f'{flags["model_path"]}train_state-{step}.pt')

        # Complete training.
        if (step >= flags["ntraining_steps"]):
          not_reached_nsteps = False
          break

        step += 1
  
  # If the run is interrupted with ctrl+C, save the last model and train state.
  # TO DO capire
  except KeyboardInterrupt:
    pass

  if rank == 0 or device == torch.device("cpu"):
    if device == torch.device("cpu"):
      simulator.save(flags["model_path"] + 'model-'+str(step)+'.pt')
    else:
      simulator.module.save(flags["model_path"] + 'model-'+str(step)+'.pt')
    train_state = dict(optimizer_state=optimizer.state_dict(),
                       global_train_state={"step": step},
                       loss=loss.item())
    torch.save(train_state, f'{flags["model_path"]}train_state-{step}.pt')

  if torch.cuda.is_available():
    distribute.cleanup()


def _get_simulator(
        metadata: json,
        connectivity_radius: float,
        vel_noise_std: float,
        device: torch.device) -> learned_simulator.LearnedSimulator:
  
  """
  Getter for the simulator.

  Args:
    metadata: JSON object with metadata.
    vel_noise_std: Velocity noise std deviation.
    device: PyTorch device 'cpu' or 'cuda'.
  """

  # Normalization stats.
  normalization_stats = {
          'mean': torch.FloatTensor(metadata['vel_mean']).to(device),
          'std': torch.sqrt(torch.FloatTensor(metadata['vel_std'])**2 +
                            vel_noise_std**2).to(device), # TO DO: perchè noise qui
      }
  
  nnode_in = 14 # TO DO: mettere automatico
  nedge_in = metadata['dim'] + 1

  # Initialize the simulator.
  simulator = learned_simulator.LearnedSimulator(
      problem_dim=metadata['dim'],
      nnode_in=nnode_in,
      nedge_in=nedge_in,
      latent_dim=128,
      nmessage_passing_steps=10,
      nmlp_layers=2,
      mlp_hidden_dim=128,
      connectivity_radius=connectivity_radius,
      boundaries=np.array(metadata['bounds']),
      normalization_stats=normalization_stats,
      dt = metadata["dt"],
      boundary_clamp_limit=metadata["boundary_augment"] if "boundary_augment" in metadata else 1.0,
      device=device)

  return simulator


def main(_):

  """
  Main function to train or evaluate the model.
  """

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if device == torch.device('cuda'):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

  myflags = {}
  myflags["data_path"] = FLAGS.data_path
  myflags["noise_std"] = FLAGS.noise_std
  myflags["lr_init"] = FLAGS.lr_init
  myflags["lr_decay"] = FLAGS.lr_decay
  myflags["lr_decay_steps"] = FLAGS.lr_decay_steps
  myflags["batch_size"] = FLAGS.batch_size
  myflags["ntraining_steps"] = FLAGS.ntraining_steps
  myflags["nsave_steps"] = FLAGS.nsave_steps
  myflags["model_file"] = FLAGS.model_file
  myflags["model_path"] = FLAGS.model_path
  myflags["train_state_file"] = FLAGS.train_state_file
  myflags["connectivity_radius"] = FLAGS.connectivity_radius

  # Train
  if FLAGS.mode == 'train':
    # If model_path does not exist create new directory.
    if not os.path.exists(FLAGS.model_path):
      os.makedirs(FLAGS.model_path)

    # Train on gpu 
    if device == torch.device('cuda'):
      world_size = torch.cuda.device_count()
      print(f"world_size = {world_size}")
      distribute.spawn_train(train, myflags, world_size, device)

    # Train on cpu  
    else:
      rank = None
      world_size = 1
      train(rank, myflags, world_size, device)

  # Evaluation
  elif FLAGS.mode in ['valid', 'test']:
    # Set device
    world_size = torch.cuda.device_count()
    if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
      device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')
    predict(device)


if __name__ == '__main__':
  app.run(main)
