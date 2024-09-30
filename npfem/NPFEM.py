import json
import os
import pickle
import glob
import re
import sys
import numpy as np

#PyTorch
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch3d.loss import chamfer_distance
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from absl import flags
from absl import app

#npfem modules
from npfem import learned_simulator
from npfem import noise_utils
from npfem import data_loader
from npfem import distribute

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Code mode definition
flags.DEFINE_enum('mode', 'train', ['train', 'valid', 'test'], help='Train, Validate or Test the model. Train is the default mode.')

# Input/output folders/files
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_string('model_path', 'models/', help=('The path for saving checkpoints of the model.'))
flags.DEFINE_string('output_path', 'outputs/', help='The path for saving outputs.')
flags.DEFINE_string('output_filename', 'test', help='Base name for saving the output.')
flags.DEFINE_bool('continue_training', False, help=('Boolean variable that is False when the training must start from the beginnig, True if it must continue from the last model generated.'))
flags.DEFINE_string('model_file', 'model.pt', help=('Model file (.pt) to be used for the prediction.'))

# Steps
flags.DEFINE_integer('n_training_steps', int(2E6), help='Total number of training steps.')
flags.DEFINE_integer('save_steps_freq', int(5000), help='Training steps frequency to save the model.')

# Hyperparameters
#flags.DEFINE_float('spatial_norm_weight', 0.1, help='Weight used to normalize spatial features') Now mesh size h is used
flags.DEFINE_float('noise_std_weight', 5, help='The std deviation weight of the noise.')
flags.DEFINE_float('velocity_scale_weight', 0.1, help='Weight used to scale velocity features')
flags.DEFINE_integer('input_velocity_steps', 5, help='Sequence length of current + previous velocities used to predict the time update.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_integer('n_message_passing_steps', int(10), help='Number of message passing steps.')

flags.DEFINE_float('boundary_clamp_limit', 1.0, help='Boundary clamp limit.') #TO DO

flags.DEFINE_integer('ar_freq', 1, help='AddRem frequence during prediction')

flags.DEFINE_float('lr_init', 1e-4, help='Initial learning rate.') #TO DO
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', int(1e6), help='Learning rate decay steps.')


# CUDA device 
flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")

# Save command line options
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
    Generate a trajectory by applying the learned model in sequence for one example in the valid/test dataset.

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
    initial_position = position[:, :FLAGS.input_velocity_steps]
    initial_velocity = velocity[:, :FLAGS.input_velocity_steps]

    # Trajectory data from the simulation.
    ground_truth_position = position[:, FLAGS.input_velocity_steps:].permute(1, 0, 2)
    ground_truth_velocity = velocity[:, FLAGS.input_velocity_steps:].permute(1, 0, 2)

    # Initialize position, velocity and cells.
    current_position = initial_position[:,-1]
    current_velocity = initial_velocity
    predicted_position = []
    predicted_velocity = []
    predicted_cells = []
    predicted_free_surf = []
    for step in range(FLAGS.input_velocity_steps):
        predicted_cells.append(cells[step, :n_cells[step], :].cpu().detach().numpy())
        predicted_free_surf.append(free_surf[step, :].cpu().detach().numpy())
    free_surf = free_surf[-1, :]

    # Cycle over the number of time steps I want to predict.
    for step in tqdm(range(nsteps+1), total=nsteps+1):
        # Get next position and velocity, each with shape (nnodes, dim), with learned simulator
        next_position, current_position, current_velocity[:,-1], next_velocity, next_cells, free_surf = simulator.learned_update(
            current_position,
            current_velocity,
            bounds = bounds,
            free_surf = free_surf,
            step = step)
        # Add the predicted next position and velocity to the trajectory vectors.
        if step > 0:
            predicted_position.append(current_position)
            predicted_velocity.append(current_velocity[:,-1])
            predicted_cells.append(next_cells)
            predicted_free_surf.append(free_surf.cpu().numpy())

        # Shift current_position/velocity, removing the oldest position in the sequence and appending the next position at the end.
        current_position = next_position
        current_velocity = torch.cat(
            [current_velocity[:, 1:], next_velocity[:, None, :]], dim=1)

    # Predicted position and velocity with shape (time, nnodes, dim).
    predicted_position = torch.stack(predicted_position)
    predicted_velocity = torch.stack(predicted_velocity)

    # Compute the error between the simulated trajectory and the predicted one.
    # TO DO: find the best way to evaluate test error
    MSE = (predicted_position - ground_truth_position) ** 2
    chamfer_dist = chamfer_distance(predicted_position, ground_truth_position) 

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

    return output_dict, MSE, chamfer_dist[0]


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
  MSE = []
  chamfer_dist = []
  with torch.no_grad(): # TO DO: understand
    for example_i, features in enumerate(ds):
      nsteps = metadata['prediction_time_steps'] - FLAGS.input_velocity_steps

      # Get data from the dataset
      position = features[0].to(device)
      velocity = features[1].to(device)
      cells =  features[2].to(device)
      bounds = features[3].to(device)
      free_surf = features[4].to(device)
      free_surf = np.squeeze(free_surf)
      start_free_surf = free_surf[0:FLAGS.input_velocity_steps, :]
      n_cells = features[5].to(device)

      print("Predicting example {}...".format(example_i))
      with torch.autocast(device_type="cuda", dtype=torch.float16):

        # Predict example rollout and evaluate example error loss
        example_output, example_MSE, example_chamfer_dist = predict_example(simulator,
                                                       position,
                                                       velocity,
                                                       cells,
                                                       n_cells,
                                                       bounds,
                                                       start_free_surf,
                                                       nsteps)

      # Print information
      print("{:<20} {:^20}".format("Chamfer distance", example_chamfer_dist))
      print("{:<20} {:^20}".format("MSE", example_MSE.mean()))

      # Save example error
      MSE.append(torch.flatten(example_MSE))
      chamfer_dist.append(example_chamfer_dist)

      # Save predicted trajectory during test 
      if FLAGS.mode == 'test':
        example_output['metadata'] = metadata # TO DO
        example_output['chamfer_distance'] = example_chamfer_dist
        example_output['MSE'] = example_MSE
        filename = f'{FLAGS.output_filename}_ex{example_i}.pkl'
        filename = os.path.join(FLAGS.output_path, filename)
        with open(filename, 'wb') as f:
          pickle.dump(example_output, f)

  print("Dataset: Chamfer distance {}, ".format(torch.mean(torch.stack(chamfer_dist))), torch.mean(torch.cat(MSE)))

  if FLAGS.mode == 'valid':
    with open("loss.txt", "a+") as file:
      file.seek(0)
      file.write(str(torch.mean(torch.stack(chamfer_dist)).detach().cpu().numpy()))
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

def velocity_loss(pred_vel, target_vel):
  """
  Compute the loss between predicted and target velocities (mean of the error over all the particles).

  Args:
    pred_velocity: Predicted velocities.
    target_velocity: Target velocities.
  """
  loss = (pred_vel - target_vel) ** 2
  loss = loss.sum(dim=-1)
  loss = loss.sum() / target_vel.shape[0]
  return loss

def save_model(simulator, FLAGS, step, optimizer, loss):
  """Save model state
  TO DO
  Args:
    rank: local rank
    device: torch device type
    simulator: Trained simulator if not will undergo training.
    flags: flags
    step: step
    epoch: epoch
    optimizer: optimizer
    train_loss: training loss at current step
    valid_loss: validation loss at current step
    train_loss_hist: training loss history at each epoch
    valid_loss_hist: validation loss history at each epoch
  """
  simulator.module.save(FLAGS.model_path  + 'model-'+str(step)+'.pt')
  train_state = dict(optimizer_state=optimizer.state_dict(),
                     global_train_state={"step": step},
                     loss=loss.item())
  torch.save(train_state, f'{FLAGS.model_path}train_state-{step}.pt')

def train(rank, world_size, device):

  """
  Train the model.

  Args:
    rank: local rank
    world_size: total number of ranks
    device: torch device type
  """

  # Initialize CUDA device
  distribute.setup(rank, world_size, device)
  device_id = rank
  print(f"cuda = {torch.cuda.is_available()}")

  # Read metadata
  with open(os.path.join(FLAGS.data_path, "metadata.json"), 'rt') as fp:
      metadata = json.loads(fp.read())

  # Get simulator and optimizer
  serial_simulator = _get_simulator(metadata, rank)
  simulator = DDP(serial_simulator.to(rank), device_ids=[rank], output_device=rank)
  optimizer = torch.optim.Adam(simulator.parameters(), lr=FLAGS.lr_init*world_size)
  # TO DO other optimizer
  #optimizer = torch.optim.RMSprop(simulator.parameters(), lr=FLAGS.lr_init*world_size)

  # Initialize the step indices
  step = 0
  epoch = 0
  steps_per_epoch = 0
  epoch_train_loss = 0

  # If continue flag is True, check if it is possible to continue the training # TO DO check
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
    simulator.module.load(FLAGS.model_path  + model_file)

    # Load train state
    train_state = torch.load(FLAGS.model_path  + train_state_file)

    # Set optimizer state
    optimizer = torch.optim.Adam(simulator.parameters())
    optimizer.load_state_dict(train_state["optimizer_state"])
    optimizer_to(optimizer, device_id)

    # Update the step index
    step = train_state["global_train_state"].pop("step")

  simulator.train()
  simulator.to(device_id)

  # Get data loader
  dl = distribute.get_data_distributed_dataloader_by_samples(f'{FLAGS.data_path}train.npz',
                                                             FLAGS.input_velocity_steps,
                                                             FLAGS.batch_size)
  # Initialize the scaler
  scaler = GradScaler()
  
  # Loop over training steps
  try:
    while step < FLAGS.n_training_steps:
      # Loop over... TO DO
      for example in dl:  
        steps_per_epoch += 1

        # Take data from DataLoader dl
        # ((position, velocity, ..., n_particles_per_example), labels) are in dl
        position = example[0][0].to(device_id)
        velocity = example[0][1].to(device_id)
        edges = example[0][2].to(device_id)
        bounds = example[0][4].to(device_id)
        n_particles_per_example = example[0][5].to(device_id)
        n_edges_per_example = example[0][6].to(device_id)
        # TO DO controllare, non dovrebbe essere necessario
        if(n_particles_per_example.size()[0]!=FLAGS.batch_size):
          continue
        labels = example[1].to(device_id)
        n_particles_per_example.to(device_id)
        #free_surf = example[0][3].to(device_id)
        #free_surf = torch.squeeze(free_surf)
        
        # Sample the noise to add to the inputs
        sampled_pos_noise, sampled_vel_noise = noise_utils.get_random_walk_noise_for_position_and_velocity_sequence(
          position, velocity, FLAGS.noise_std_weight, metadata['dt'])
        sampled_pos_noise = sampled_pos_noise.to(device_id)
        sampled_vel_noise = sampled_vel_noise.to(device_id)
        
        noisy_position = position + sampled_pos_noise
        noisy_velocity = velocity + sampled_vel_noise

        # Zero the parameter gradients
        optimizer.zero_grad(set_to_none=True)

        # Automatic mixed precision
        with torch.autocast(device_type="cuda", dtype=torch.float16):

            # Get the velocity predicted with GNN and the target velocity from data
            pred_vel, target_vel = simulator.module.predict_velocity(
                next_velocity=labels.to(rank),
                current_position=noisy_position.to(rank),
                velocity_sequence=noisy_velocity.to(rank),
                edges=edges.to(rank),
                bounds=bounds.to(rank),
                n_particles_per_example=n_particles_per_example.to(rank),
                n_edges_per_example=n_edges_per_example.to(rank)
            )
        
            # Calculate the loss (the mean of the velocity error over all the particles TO DO funzione
            loss = velocity_loss(pred_vel, target_vel)
            epoch_train_loss += loss
        
        # TO DO
        # Computes the gradient of loss
        #loss.backward()
        #optimizer.step()
        scaler.scale(loss).backward()         
        scaler.step(optimizer)
        scaler.update()

        # Update learning rate TO DO
        #lr_new = FLAGS.lr_init  #* (FLAGS.lr_decay ** (step/FLAGS.lr_decay_steps)) # TO DO 
        #for param in optimizer.param_groups:
        #  param['lr'] = lr_new
        print(f'Training step: {step}/{FLAGS.n_training_steps}. Loss: {loss}.', flush=True)
   
        # Save model and train state
        if rank == 0 and step % FLAGS.save_steps_freq == 0:
            save_model(simulator, FLAGS, step, optimizer, loss)
        
        # Update step number
        step += 1
        
        # Complete training
        if (step >= FLAGS.n_training_steps):
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
      if (step >= FLAGS.n_training_steps):
          break 

  # If the run is interrupted with ctrl+C, save the last model and train state
  except KeyboardInterrupt:
    pass
  
  # save model and train state
  if rank == 0:
    save_model(simulator, FLAGS, step, optimizer, loss)


def _get_simulator(
        metadata: json,
        device: torch.device) -> learned_simulator.LearnedSimulator:
  
  """Instantiates the simulator.

  Args:
    metadata: JSON object with metadata.
    device: PyTorch device (cuda).
  """

  # Normalization stats
  normalization_stats = {
          'mean': torch.FloatTensor(metadata['vel_mean']).to(device),
          'std': torch.sqrt(torch.FloatTensor(metadata['vel_std'])**2+6.7e-4**2).to(device), 
      }
  
  #nnode_in = (current + previous C velocities)*2dim + distance from boundaries 
  nnode_in = (FLAGS.input_velocity_steps)*2 + 4 #+ 1 # TO DO: bound e free surf on off
  nedge_in = 3

  # Init simulator.
  simulator = learned_simulator.LearnedSimulator(
      nnode_in=nnode_in,
      nedge_in=nedge_in,
      latent_dim=128,
      n_message_passing_steps=FLAGS.n_message_passing_steps,
      nmlp_layers=2,
      mlp_hidden_dim=128,
      boundary=torch.Tensor(metadata['bounds']),
      normalization_stats=normalization_stats,
      dt = metadata['dt'],
      h = metadata['h'],
      velocity_scale_weight=FLAGS.velocity_scale_weight,
      boundary_clamp_limit=FLAGS.boundary_clamp_limit,
      ar_freq = FLAGS.ar_freq,
      device=device)

  return simulator


def main(_):

  """Train or evaluate the model.
  """

  device = torch.device('cuda')
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "29500"
  
  # Train
  if FLAGS.mode == 'train':
    # If model_path does not exist create new directory.
    if not os.path.exists(FLAGS.model_path):
      os.makedirs(FLAGS.model_path)

    world_size = torch.cuda.device_count()
    print(f"world_size = {world_size}")
    distribute.spawn_train(train, world_size, device)

  # Evaluation
  elif FLAGS.mode in ['valid', 'test']:
    predict(device)


if __name__ == '__main__':
  app.run(main)

# Time measure
#start = torch.cuda.Event(enable_timing=True)
#end = torch.cuda.Event(enable_timing=True)
#torch.cuda.synchronize()
#start.record()
#torch.cuda.synchronize()
#end.record()
#torch.cuda.synchronize()
#print(start.elapsed_time(end))