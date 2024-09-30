"""
Rendering of the simulator output as a vtk file.
"""

import pickle

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import os
from pyevtk.hl import pointsToVTK

flags.DEFINE_string("output_path", None, help="Directory where output.pkl are located.")
flags.DEFINE_string("output_file", None, help="Name of output .pkl file.")

FLAGS = flags.FLAGS

class VTKwriter():

    """
    Render output data into vtk files.
    """

    def __init__(self, input_dir, input_name):

        """
            Initialize VTKwriter class.

        Args:
            input_dir (str): Directory where rollout.pkl are located.
            input_name (str): Name of rollout .pkl file.
        """

        self.input_dir = input_dir
        self.input_name = input_name
        self.output_dir = input_dir
        self.output_name = input_name

        # Get trajectory.
        with open(f"{self.input_dir}{self.input_name}.pkl", "rb") as file:
            output_data = pickle.load(file)
        self.output_data = output_data

        # Initialize dicts.
        trajectory = {}
        velocity = {}
        cells = {}
        trajectory = np.concatenate([output_data["initial_position"], output_data["predicted_position"]], axis=0)
        velocity = np.concatenate([output_data["initial_velocity"], output_data["predicted_velocity"]], axis=0)
        cells = output_data["predicted_cells"]
        free_surf = output_data["predicted_free_surf"]
        self.trajectory = trajectory
        self.velocity = velocity
        self.cells = cells
        self.free_surf = free_surf

        # Trajectory information.
        self.dims = trajectory.shape[2]
        self.num_particles = trajectory.shape[1]
        self.num_steps = trajectory.shape[0]
        self.boundaries = output_data["metadata"]["bounds"]

    def write_vtk(self):
        """
        Write `.vtk` files for each timestep for each rollout case.
        """

        path = f"{self.output_dir}{self.output_name}_vtk"
        if not os.path.exists(path):
           os.makedirs(path)

        # Loop over timesteps.
        for i, coord in enumerate(self.trajectory):
            filename = f"{path}/test.vtk.{i}"
            with open(filename, 'w') as file:
                
                # Write intro.
                file.write("# vtk DataFile Version 1.0\n2D Unstructured Grid of Linear Triangles\nASCII\n\nDATASET UNSTRUCTURED_GRID\n")
                
                # Write points.
                file.write(f"POINTS \t {coord.shape[0]} float\n")
                pos=np.hstack((coord, (np.full((coord.shape[0],1),0))))
                np.savetxt(file, pos, delimiter='\t', fmt='%.6f')
                file.write("\n")
                
                # Write cells
                file.write(f"CELLS \t {self.cells[i].shape[0]} \t {self.cells[i].shape[0]*4}\n")
                cell=np.hstack(((np.full((self.cells[i].shape[0],1),3)), self.cells[i]))
                np.savetxt(file, cell, delimiter='\t', fmt='%d')
                file.write("\n")
                file.write(f"CELL_TYPES \t {self.cells[i].shape[0]}\n")        
                cell_type=np.full((self.cells[i].shape[0],1),5)
                np.savetxt(file, cell_type, delimiter='\t', fmt='%d')
                
                # Write velocity
                file.write(f"POINT_DATA \t {coord.shape[0]} \n")     
                file.write("VECTORS Velocity float\n")
                vel=np.hstack((self.velocity[i], (np.full((self.velocity[i].shape[0],1),0))))
                np.savetxt(file, vel, delimiter='\t', fmt='%.6f')
                
                # Write free surface
                file.write(f"SCALARS FreeSurf float \n")     
                file.write("LOOKUP_TABLE default\n")
                print(i)
                free_surf=self.free_surf[i]
                np.savetxt(file, free_surf, delimiter='\t', fmt='%.6f')

        print(f"vtk saved to: {self.output_dir}{self.output_name}...")

def main(_):
    if not FLAGS.output_path:
        raise ValueError("An output directory must be passed with --output_path.")
    if not FLAGS.output_file:
        raise ValueError("A output file name must be passed with --.")

    render = VTKwriter(input_dir=FLAGS.output_path, input_name=FLAGS.output_file)
    render.write_vtk()


if __name__ == '__main__':
    app.run(main)