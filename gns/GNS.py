import json
import os
import pickle
import glob
import re
import sys
import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from absl import flags
from absl import app

from gns import learned_simulator
from gns import noise_utils
from gns import data_loader

import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Code modality definition
flags.DEFINE_enum(
    'mode', 'train', ['train', 'valid', 'test'],
    help='Train model, validation or test')

# Input/output folders/files
flags.DEFINE_string('data_path', None, help='The dataset directory')
flags.DEFINE_string('model_path', 'models/', help=('The path for saving checkpoints of the model'))
flags.DEFINE_string('output_path', 'outputs/', help='The path for saving outputs')
flags.DEFINE_string('output_filename', 'test', help='Base name for saving the output')
flags.DEFINE_bool('continue_training', False, help=('Boolean variable that is False when the training must start from the beginnig, True if it must continue from the last model generated'))
flags.DEFINE_string('model_file', 'model.pt', help=('Model file (.pt) to use for the prediction'))

# Steps
flags.DEFINE_integer('ntraining_steps', int(2E7), help='Number of training steps.')
flags.DEFINE_integer('nsave_steps', int(5000), help='Nsteps frequency to save the model')

# Hyperparameters
flags.DEFINE_integer('batch_size', 2, help='The batch size')
flags.DEFINE_float('noise_std_weight', 500, help='The std deviation weight of the noise')
flags.DEFINE_float('lr_init', 1e-4, help='Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', int(5e6), help='Learning rate decay steps.')
flags.DEFINE_integer('input_sequence_length', 6, help='Sequence length of previous velocities/positions used to predict the update.')
flags.DEFINE_integer('nmessage_passing_steps', int(10), help='Number of message passing steps.')
flags.DEFINE_float('spatial_norm_weight', 0.1, help='Weight used to normalize spatial features')
flags.DEFINE_float('boundary_clamp_limit', 1.0, help='Boundary clamp limit.')
flags.DEFINE_integer('ar_freq', 1000, help='AddRem frequence during prediction')

# CUDA device 
flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

def predict_example(
    simulator: learned_simulator.LearnedSimulator,
    position: torch.tensor,
    velocity: torch.tensor,
    cells: torch.tensor,
    n_cells: torch.tensor,
    bounds: torch.tensor,
    free_surf: torch.tensor,
    nsteps: int):

    """
    Generate a trajectory by applying the learned model in sequence for an example in the valid/test dataset.

    Args:
        simulator: Learned simulator.
        position: Positions of particles. Shape: (timesteps, n_particles, ndims).
        velocity: Velocities of particles. Shape: (timesteps, n_particles, ndims).
        cells: Cells of the PFEM solution. Shape: (timesteps, D, 3).
        n_cells: Number of cells at each step in the PFEM solution. Shape(timesteps).
        bounds: Vector indicating boundary particles. Shape: (n_particles).
        free_surf: Vector indicating free surface particles. Shape: (n_particles).
        nsteps: Number of steps.
    """

    # Initial configuration.
    initial_position = position[:, :FLAGS.input_sequence_length]
    initial_velocity = velocity[:, :FLAGS.input_sequence_length]

    filt_initial_position = initial_position
    filt_initial_velocity = initial_velocity

    # Trajectory data from the simulation.
    ground_truth_position = position[:, FLAGS.input_sequence_length:].permute(1, 0, 2)
    ground_truth_velocity = velocity[:, FLAGS.input_sequence_length:].permute(1, 0, 2)

    # Initialize position, velocity and cells.
    current_position = initial_position
    current_velocity = initial_velocity
    predicted_position = []
    predicted_velocity = []
    predicted_cells = []
    predicted_free_surf = []
    for step in range(FLAGS.input_sequence_length):
        predicted_cells.append(cells[step, :n_cells[step], :].cpu().detach().numpy())
        predicted_free_surf.append(free_surf[step, :].cpu().detach().numpy())
    free_surf = free_surf[-1, :]
    free_surf[bounds==1] = 0

    # Cycle over the number of time steps I want to predict.
    for step in tqdm(range(nsteps), total=nsteps):
        # Get next position and velocity, each with shape (nnodes, dim), with learned simulator
        next_position, current_position[:,-1], current_velocity[:,-1], next_velocity, next_cells, free_surf = simulator.learned_update(
            current_position,
            current_velocity,
            bounds = bounds,
            free_surf = free_surf,
            step = step)
        # Add the predicted next position and velocity to the trajectory vectors.
        free_surf[bounds==1] = 0
        predicted_position.append(torch.where(torch.stack((bounds, bounds), dim=1), initial_position[:,0], next_position))
        predicted_velocity.append(torch.where(torch.stack((bounds, bounds), dim=1), initial_velocity[:,0], next_velocity))
        predicted_cells.append(next_cells)
        predicted_free_surf.append(free_surf.cpu().numpy())

        # Shift current_position/velocity, removing the oldest position in the sequence and appending the next position at the end.
        current_position = torch.cat(
            [current_position[:, 1:], next_position[:, None, :]], dim=1)
        current_velocity = torch.cat(
            [current_velocity[:, 1:], next_velocity[:, None, :]], dim=1)

    # Predicted position and velocity with shape (time, nnodes, dim).
    predicted_position = torch.stack(predicted_position)
    predicted_velocity = torch.stack(predicted_velocity)

    # Compute the error between the simulated trajectory and the predicted one.
    # TO DO: find the best way to evaluate test error
    example_error = (predicted_position - ground_truth_position) ** 2
    example_error2 = chamfer_distance(predicted_position, ground_truth_position) 

    # Output data structure
    output_dict = {
        'initial_position': initial_position.permute(1, 0, 2).cpu().numpy(),
        'initial_velocity': initial_velocity.permute(1, 0, 2).cpu().numpy(),
        'predicted_position': predicted_position.cpu().numpy(),
        'predicted_velocity': predicted_velocity.cpu().numpy(),
        'predicted_cells': predicted_cells,
        'predicted_free_surf': predicted_free_surf,
        'ground_truth_position': ground_truth_position.cpu().numpy(),
        'ground_truth_velocity': ground_truth_velocity.cpu().numpy()
    }

    return output_dict, example_error, example_error2[0]


def predict(device: str):

  """Predict and evaluate trajectories for valid/test datasets.
  """

  # Read metadata
  with open(os.path.join(FLAGS.data_path, "metadata.json"), 'rt') as fp:
      metadata = json.loads(fp.read())

  # Get the valid or test dataset
  split = 'test' if FLAGS.mode == 'test' else 'valid'
  ds = data_loader.get_data_loader_by_trajectories(path=f"{FLAGS.data_path}{split}.npz")

  # Define output path
  if not os.path.exists(FLAGS.output_path):
    os.makedirs(FLAGS.output_path)

  # Get learned simulator
  simulator = _get_simulator(metadata, device)
  
  if os.path.exists(FLAGS.model_path + FLAGS.model_file):
    simulator.load(FLAGS.model_path + FLAGS.model_file)
  else:
    raise Exception(f"Model does not exist at {FLAGS.model_path + FLAGS.model_file}")

  simulator.to(device)
  simulator.eval()

  # Loop over the examples in the dataset
  error = []
  error2 = []
  with torch.no_grad(): # TO DO: understand
    for example_i, features in enumerate(ds):
      nsteps = metadata['sequence_length'] - FLAGS.input_sequence_length

      # Get data from the dataset
      position = features[0].to(device)
      velocity = features[1].to(device)
      cells =  features[2].to(device)
      bounds = features[3].to(device)
      free_surf = features[4].to(device)
      free_surf = np.squeeze(free_surf)
      start_free_surf = free_surf[0:FLAGS.input_sequence_length, :]
      n_cells = features[5].to(device)

      with torch.autocast(device_type="cuda", dtype=torch.float16):

        # Predict example rollout and evaluate example error loss
        example_rollout, example_error, example_error2 = predict_example(simulator,
                                                       position,
                                                       velocity,
                                                       cells,
                                                       n_cells,
                                                       bounds,
                                                       start_free_surf,
                                                       nsteps)

      # Print information
      print("Predicting example {} error: {}".format(example_i, example_error.mean()))
      print("Predicting example {} error: {}".format(example_i, example_error2))

      # Save example error
      error.append(torch.flatten(example_error))
      error2.append(example_error2)

      # Save predicted trajectory during test 
      if FLAGS.mode == 'test':
        example_rollout['metadata'] = metadata # TO DO
        example_rollout['loss'] = example_error
        filename = f'{FLAGS.output_filename}_ex{example_i}.pkl'
        filename = os.path.join(FLAGS.output_path, filename)
        with open(filename, 'wb') as f:
          pickle.dump(example_rollout, f)

  print("Mean error prediction dataset: {}".format(torch.mean(torch.cat(error))))
  print("Mean error prediction dataset: {}".format(torch.mean(torch.stack(error2))))
  # Uncomment the following lines when, in the validation process, you are interested
  # in running this code for every model generated during training keeping track of the 
  # error history to find the best one and to evaluate the training process
 
  if FLAGS.mode == 'valid':
    with open("loss.txt", "a+") as file:
      file.seek(0)
      file.write(str(torch.mean(torch.stack(error2)).detach().cpu().numpy()))
      file.write("\n")


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


def train(rank, world_size, device):

  """
  Train the model.

  Args:
    rank: local rank
    world_size: total number of ranks
    device: torch device type
  """

  # Initialize CUDA device
  device_id = device
  print(f"cuda = {torch.cuda.is_available()}")

  # Read metadata
  with open(os.path.join(FLAGS.data_path, "metadata.json"), 'rt') as fp:
      metadata = json.loads(fp.read())
  # Get simulator and optimizer
  simulator = _get_simulator(metadata, device)
  #optimizer = torch.optim.RMSprop(simulator.parameters(), lr=FLAGS.lr_init*world_size)
  optimizer = torch.optim.Adam(simulator.parameters(), lr=FLAGS.lr_init*world_size)
  # Initialize the step indices
  step = 0
  epoch = 0
  steps_per_epoch = 0
  epoch_train_loss = 0

  # If continue flag is True, check if it is possible to continue the training # TO DO 
  if FLAGS.continue_training:
    # Search for the last model generated, assumes model and train_state files are in step.
    fnames = glob.glob(f'{FLAGS.model_path}*model*pt')
    max_model_number = 0
    expr = re.compile(".*model-(\d+).pt")
    for fname in fnames:
        model_num = int(expr.search(fname).groups()[0])
        if model_num > max_model_number:
            max_model_number = model_num
    # reset names to point to the latest.
    model_file = f"model-{max_model_number}.pt"
    train_state_file = f"train_state-{max_model_number}.pt"

    # Load model
    simulator.load(FLAGS.model_path  + model_file)

    # Load train state
    train_state = torch.load(FLAGS.model_path  + train_state_file)

    # Set optimizer state
    optimizer = torch.optim.Adam(
    simulator.parameters())
    optimizer.load_state_dict(train_state["optimizer_state"])
    optimizer_to(optimizer, device_id)
    '''
    optimizer = torch.optim.RMSprop(
    simulator.parameters())
    optimizer.load_state_dict(train_state["optimizer_state"])
    optimizer_to(optimizer, device_id)
    '''
    # Update the step index
    step = train_state["global_train_state"].pop("step")

  # TO DO: understand
  simulator.train()
  simulator.to(device_id)
  # Get data loader
  get_data_loader = data_loader.get_data_loader_by_samples
  # Load training data
  dl = get_data_loader(
      path=f'{FLAGS.data_path}train.npz',
      input_length_sequence=FLAGS.input_sequence_length,
      batch_size=FLAGS.batch_size,
  ) # TO DO drop last dovrebbe sostituire if(n_particles_per_example.size()[0]!=FLAGS.batch_size):  continue
  scaler = GradScaler()
  # Loop over training steps
  try:
    while step < FLAGS.ntraining_steps:

      # Loop over... TO DO
      for example in dl:  
        steps_per_epoch += 1
        
        # ((position, velocity, ..., n_particles_per_example), labels) are in dl
        position = example[0][0].to(device_id)
        velocity = example[0][1].to(device_id)
        edges = example[0][2].to(device_id)
        free_surf = example[0][3].to(device_id)
        bounds = example[0][4].to(device_id)
        free_surf = torch.squeeze(free_surf)
        free_surf[bounds==1] = 0
        n_particles_per_example = example[0][5].to(device_id)
        n_edges_per_example = example[0][6].to(device_id)
        if(n_particles_per_example.size()[0]!=FLAGS.batch_size):
          continue
        labels = example[1].to(device_id)
        n_particles_per_example.to(device_id)
        # Sample the noise to add to the inputs to the model during training
        # TO DO: capire errore migliore, valore, applicato a position o velocity...
        sampled_pos_noise, sampled_vel_noise = noise_utils.get_random_walk_noise_for_position_and_velocity_sequence(
          position, velocity, noise_std_weight=FLAGS.noise_std_weight)
        sampled_pos_noise = sampled_pos_noise.to(device_id)
        sampled_vel_noise = sampled_vel_noise.to(device_id)
        sampled_pos_noise[bounds==1] = 0 
        sampled_vel_noise[bounds==1] = 0 

        noisy_position = position #+ sampled_pos_noise
        noisy_velocity = velocity + sampled_vel_noise

        optimizer.zero_grad()
        #with torch.autocast(device_type="cuda"):
        with torch.autocast(device_type="cuda", dtype=torch.float16):

            # Get the velocity predicted with GNN and the target velocity from data
            device_or_rank = device
            pred_vel, target_vel = simulator.predict_velocity(
                next_velocity=labels,
                position_sequence=noisy_position,
                velocity_sequence=noisy_velocity,
                edges=edges,
                free_surf=free_surf,
                bounds=bounds,
                n_particles_per_example=n_particles_per_example,
                n_edges_per_example=n_edges_per_example
            )
            # Calculate the loss (the mean of the velocity error over all the particles TO DO funzione
            #pred_vel = pred_vel#[~bounds] TO DO provare
            #target_vel = target_vel#[~bounds]
            loss = (pred_vel - target_vel) ** 2
            #mean_abs_vel = torch.mean(torch.abs(target_vel))
            #loss = loss / (mean_abs_vel ** 2) TO DO provare
            loss = loss.sum(dim=-1)
            loss = loss.sum() / target_vel.shape[0]
        
            epoch_train_loss += loss
        
        # Computes the gradient of loss
        #loss.backward()
        #optimizer.step()
        
        scaler.scale(loss).backward()         
        scaler.step(optimizer)
        scaler.update()
        # Update learning rate
        lr_new = FLAGS.lr_init  #* (FLAGS.lr_decay ** (step/FLAGS.lr_decay_steps))

        for param in optimizer.param_groups:
          param['lr'] = lr_new

        print(f'Training step: {step}/{FLAGS.ntraining_steps}. Loss: {loss}.')
        # Save model state
        if step % FLAGS.nsave_steps == 0:
            simulator.save(FLAGS.model_path  + 'model-'+str(step)+'.pt')
            train_state = dict(optimizer_state=optimizer.state_dict(),
                               global_train_state={"step": step},
                               loss=loss.item())
            torch.save(train_state, f'{FLAGS.model_path}train_state-{step}.pt')
        

        step += 1
        # Complete training
        if (step >= FLAGS.ntraining_steps):
          break

      # Epoch level statistics
      # Training loss at epoch
      epoch_train_loss /= steps_per_epoch
      epoch_train_loss = torch.tensor([epoch_train_loss]).to(device_id)
      print(f'Epoch {epoch}, training loss: {epoch_train_loss.item()}')
  
      # Reset epoch training loss
      epoch_train_loss = 0
      if steps_per_epoch >= len(dl):
          epoch += 1
          steps_per_epoch = 0
      
      # Complete training
      if (step >= FLAGS.ntraining_steps):
          break 

  # If the run is interrupted with ctrl+C, save the last model and train state
  # TO DO capire
  except KeyboardInterrupt:
    pass

  # TO DO mettere in una funzione 
  if rank == 0 or device == torch.device("cpu"):
    simulator.save(FLAGS.model_path + 'model-'+str(step)+'.pt')
    train_state = dict(optimizer_state=optimizer.state_dict(),
                       global_train_state={"step": step},
                       loss=loss.item())
    torch.save(train_state, f'{FLAGS.model_path}train_state-{step}.pt')


def _get_simulator(
        metadata: json,
        device: torch.device) -> learned_simulator.LearnedSimulator:
  
  """Instantiates the simulator.

  Args:
    metadata: JSON object with metadata.
    device: PyTorch device 'cpu' or 'cuda'.
  """

  # Normalization stats
  normalization_stats = {
          'mean': torch.FloatTensor(metadata['vel_mean']).to(device),
          'std': torch.sqrt(torch.FloatTensor(metadata['vel_std'])**2+6.7e-4**2).to(device), 
      }
  
  nnode_in = (FLAGS.input_sequence_length-1)*2 + 4 + 1 # TO DO: bound e free surf on off
  nedge_in = 3

  # Init simulator.
  simulator = learned_simulator.LearnedSimulator(
      nnode_in=nnode_in,
      nedge_in=nedge_in,
      latent_dim=128,
      nmessage_passing_steps=FLAGS.nmessage_passing_steps,
      nmlp_layers=2,
      mlp_hidden_dim=128,
      boundary=torch.Tensor(metadata['bounds']),
      normalization_stats=normalization_stats,
      dt = metadata["dt"],
      spatial_norm_weight=FLAGS.spatial_norm_weight,
      boundary_clamp_limit=FLAGS.boundary_clamp_limit,
      ar_freq = FLAGS.ar_freq,
      device=device)

  return simulator


def main(_):

  """Train or evaluates the model.
  """

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if device == torch.device('cuda'):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
  # Train
  if FLAGS.mode == 'train':
    # If model_path does not exist create new directory.
    if not os.path.exists(FLAGS.model_path):
      os.makedirs(FLAGS.model_path)

    rank = None
    world_size = 1
    train(rank, world_size, device)

  # Evaluation
  elif FLAGS.mode in ['valid', 'test']:
    # Set device
    world_size = torch.cuda.device_count()
    predict(device)


if __name__ == '__main__':
  app.run(main)
