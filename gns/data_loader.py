import torch
import numpy as np

def load_npz_data(path):

    """
    Load data stored in npz format.

    The file format for Python 3.9 or less supports ragged arrays and Python 3.10
    requires a structured array. This function supports both formats.

    Args:
        path (str): Path to npz file.

    Returns:
        data (list): List of tuples of the form (position, velocity, moved_particle).
    """

    with np.load(path, allow_pickle=True) as data_file:
        if 'gns_data' in data_file:
            data = data_file['gns_data']
        else:
            data = [item for _, item in data_file.items()]
    return data


class SamplesDataset(torch.utils.data.Dataset):

    """
    Dataset of samples of trajectories used for training.
    
    Each sample is a tuple of the form (position, velocity, moved_particle).
    position is a numpy array of shape (sequence_length, n_particles, dimension).
    velocity is a numpy array of shape (sequence_length, n_particles, dimension).
    particle_move is a numpy array of shape (sequence_length, n_particles, dimension)
        which tells if a particle was subject to a non-physical move by the PFEM algorithm during a time-step.

    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input sequence.

    Attributes:
        _data (list): List of tuples of the form (position, velocity, particle_move).
        _dimension (int): Dimension of the data.
        _input_length_sequence (int): Length of input sequence.
        _data_lengths (list): List of lengths of trajectories in the dataset.
        _length (int): Total number of samples in the dataset.
        _precompute_cumlengths (np.array): Precomputed cumulative lengths of trajectories in the dataset.
    """

    def __init__(self, path, input_length_sequence):
        super().__init__()
        # Data are loaded as list of tuples of the form (position, velocity, moved_particle).
        self._data = load_npz_data(path)

        #Length of each trajectory in the dataset excluding the input_length_sequence. May be variable between data
        self._dimension = self._data[0][0].shape[-1]
        self._input_length_sequence = input_length_sequence
        self._data_lengths = [x.shape[0] - self._input_length_sequence for x, _, _, _, _, _, in self._data]
        self._length = sum(self._data_lengths)

        # Pre-compute cumulative lengths to allow fast indexing in __getitem__
        self._precompute_cumlengths = [sum(self._data_lengths[:x]) for x in range(1, len(self._data_lengths) + 1)]
        self._precompute_cumlengths = np.array(self._precompute_cumlengths, dtype=int)

    def __len__(self):

        """
        Return length of dataset.
        
        Returns:
            int: Length of dataset.
        """

        return self._length

    def __getitem__(self, idx):

        """
        Returns a training example from the dataset.
        
        Args:
            idx (int): Index of training example.

        Returns:
            tuple: Tuple of the form ((position, velocity, particle_move, n_particles_per_example), next_velocity).
        """

        # Select the trajectory.
        trajectory_idx = np.searchsorted(self._precompute_cumlengths - 1, idx, side="left")

        # Compute index of pick along time-dimension of trajectory.
        start_of_selected_trajectory = self._precompute_cumlengths[trajectory_idx - 1] if trajectory_idx != 0 else 0
        time_idx = self._input_length_sequence + (idx - start_of_selected_trajectory)

        # Prepare training features with the rigth shape.
        position = self._data[trajectory_idx][0][time_idx - self._input_length_sequence:time_idx]
        position = np.transpose(position, (1, 0, 2))  # (nparticles, input_sequence_length, dimension)
        velocity = self._data[trajectory_idx][1][time_idx - self._input_length_sequence:time_idx]
        velocity = np.transpose(velocity, (1, 0, 2)) # (nparticles, input_sequence_length, dimension)
        n_cells = np.transpose(self._data[trajectory_idx][2])
        n_cells = n_cells[time_idx]
        cells = self._data[trajectory_idx][3][time_idx]
        cells = cells[:n_cells, :]
        free_surf = np.transpose(self._data[trajectory_idx][4][time_idx])
        bound =  np.transpose(self._data[trajectory_idx][5])
        #particle_move = np.transpose(particle_move) # (nparticles, input_sequence_length)
        n_particles_per_example = position.shape[0] # scalar
        n_cells_per_example = cells.shape[0]
        # Training label: next step velocity
        label = self._data[trajectory_idx][1][time_idx]

        # Training example: ((features), label)
        training_example = ((position, velocity, cells, free_surf, bound, n_particles_per_example, n_cells_per_example), label)

        return training_example

def collate_fn(data):

    """
    Collate function for SamplesDataset.

    Args:
        data: List of tuples of numpy arrays of the form ((features), label).
              Length of the list = batch size

    Returns:
        tuple: Tuple of the form ((features), label).
               features and labels ar torch tensor where data from batch are stored contiguously
    """
 
    position_list = []
    velocity_list = []
    cells_list = []
    n_particles_per_example_list = []
    n_cells_per_example_list = []
    free_list = []
    bound_list = []
    label_list = []

    for ((positions, velocities, cells, free_surf, bound, n_particles_per_example, n_cells_per_example), label) in data:
        position_list.append(positions)
        velocity_list.append(velocities)
        cells_list.append(cells)
        free_list.append(free_surf)
        bound_list.append(bound)
        n_particles_per_example_list.append(n_particles_per_example)
        n_cells_per_example_list.append(n_cells_per_example)
        label_list.append(label)

    collated_data = (
        (
            torch.tensor(np.vstack(position_list)).to(torch.float32).contiguous(),
            torch.tensor(np.vstack(velocity_list)).to(torch.float32).contiguous(),
            torch.tensor(np.vstack(cells_list)).to(torch.int).contiguous(),
            torch.tensor(np.vstack(free_list)).to(torch.bool).contiguous(),
            torch.tensor(np.hstack(bound_list)).to(torch.bool).contiguous(),
            torch.tensor(n_particles_per_example_list).contiguous(),
            torch.tensor(n_cells_per_example_list).contiguous(),
        ),
        torch.tensor(np.vstack(label_list)).to(torch.float32).contiguous()
    )

    return collated_data


class TrajectoriesDataset(torch.utils.data.Dataset):

    """
    Dataset of trajectories for valid/test.

    Each trajectory is a tuple of the form (position, velocity, moved_particle)
    position is a numpy array of shape (sequence_length, n_particles, dimension)
    velocity is a numpy array of shape (sequence_length, n_particles, dimension)
    particle_move is a numpy array of shape (sequence_length, n_particles, dimension)
        which tells if a particle was subject to a non-physical move by the PFEM algorithm during a time-step
    """

    def __init__(self, path):
        super().__init__()

        # Data are loaded as list of tuples of the form (position, velocity, moved_particle)
        self._data = load_npz_data(path)
        self._dimension = self._data[0][0].shape[-1]
        self._length = len(self._data)

    def __len__(self):

        """
        Return length of dataset.

        Returns:
            int: Length of dataset.
        """

        return self._length

    def __getitem__(self, idx):

        """
        Returns a valid/test trajectory from the dataset.

        Args:
            idx (int): Index of training example.

        Returns:
            tuple: Tuple named,
              trajectory = (position, velocity, particle_move, n_particles_per_example).
        """

        position, velocity, n_cells, cells, free_surf, bounds = self._data[idx]
        position = np.transpose(position, (1, 0, 2))
        velocity = np.transpose(velocity, (1, 0, 2))
        n_particles_per_example = position.shape[0]

        trajectory = (
            torch.tensor(position).to(torch.float32).contiguous(),
            torch.tensor(velocity).to(torch.float32).contiguous(),
            torch.tensor(cells).to(torch.int).contiguous(),
            torch.tensor(bounds).to(torch.bool).contiguous(),
            torch.tensor(free_surf).to(torch.bool).contiguous(),
            torch.tensor(n_cells).to(torch.int).contiguous(),
            n_particles_per_example
        )

        return trajectory


def get_data_loader_by_samples(path, input_length_sequence, batch_size, shuffle=True):
    
    """
    Returns a pytorch data loader for the training dataset.

    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input sequence.
        batch_size (int): Batch size.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: pytorch data loader for the dataset.
    """

    dataset = SamplesDataset(path, input_length_sequence)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16,
                                       pin_memory=True, collate_fn=collate_fn)


def get_data_loader_by_trajectories(path):
    
    """Returns a data loader for the valid/test dataset.

    Args:
        path (str): Path to dataset.

    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset.
    """

    dataset = TrajectoriesDataset(path)
    return torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False,
                                       pin_memory=True)