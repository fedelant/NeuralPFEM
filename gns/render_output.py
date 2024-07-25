"""
Rendering of the simulator output as a gif or a vtk file.
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
flags.DEFINE_integer("step_stride", 3, help="Stride of steps to skip.") # TO DO 
flags.DEFINE_bool("change_yz", False, help="Change y and z axis.") 
flags.DEFINE_enum("output_mode", "gif", ["gif", "vtk"], help="Type of render output.")

FLAGS = flags.FLAGS

class Render():

    """
    Render output data into gif or vtk files.
    """

    def __init__(self, input_dir, input_name):

        """
            Initialize render class.

        Args:
            input_dir (str): Directory where rollout.pkl are located.
            input_name (str): Name of rollout .pkl file.
        """

        # Srings to describe output cases for data and render.
        output_cases = [
            [["ground_truth_position", "ground_truth_velocity"], "Reality"], [["predicted_position", "predicted_velocity"], "GNS"]]
        self.output_cases = output_cases
        self.input_dir = input_dir
        self.input_name = input_name
        self.output_dir = input_dir
        self.output_name = input_name

        # Get trajectory.
        with open(f"{self.input_dir}{self.input_name}.pkl", "rb") as file:
            output_data = pickle.load(file)
        self.output_data = output_data
        trajectory = {}
        velocity_x = {}
        velocity_y = {}
        for output_case in output_cases:
            trajectory[output_case[0][0]] = np.concatenate(
                [output_data["initial_position"], output_data[output_case[0][0]]], axis=0
            )
            velocity_x[output_case[0][1]] = np.concatenate(
                [output_data["initial_velocity"], output_data[output_case[0][1]]], axis=0
            )
            velocity_y[output_case[0][1]] = np.concatenate(
                [output_data["initial_velocity"][:,:,1], output_data[output_case[0][1]][:,:,1]], axis=0
            )

        self.trajectory = trajectory
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y

        # Trajectory information.
        self.dims = trajectory[output_cases[0][0][0]].shape[2]
        self.num_particles = trajectory[output_cases[0][0][0]].shape[1]
        self.num_steps = trajectory[output_cases[0][0][0]].shape[0]
        self.boundaries = [[-1,3],[-1, 2]]#output_data["metadata"]["bounds"]

    def render_gif_animation(
            self, point_size=1, timestep_stride=3, vertical_camera_angle=20, viewpoint_rotation=0.5, change_yz=False
    ):
        
        """
        Render .gif animation from .pkl trajectory data.

        Args:
            point_size (int): Size of particle in visualization.
            timestep_stride (int): Stride of steps to skip. 
            vertical_camera_angle (float): Vertical camera angle in degree.
            viewpoint_rotation (float): Viewpoint rotation in degree.

        Returns:
            .gif file animation.
        """

        # Init figures.
        fig = plt.figure()
        if self.dims == 2:
            ax1 = fig.add_subplot(1, 2, 1, projection='rectilinear')
            ax2 = fig.add_subplot(1, 2, 2, projection='rectilinear')
            axes = [ax1, ax2]
        elif self.dims == 3:
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            axes = [ax1, ax2]

        # Define datacase name.
        trajectory_datacases = [self.output_cases[0][0], self.output_cases[1][0]]
        render_datacases = [self.output_cases[0][1], self.output_cases[1][1]]

        # Get boundary of simulation.
        xboundary = self.boundaries[0]
        yboundary = self.boundaries[1]
        if self.dims == 3:
            zboundary = self.boundaries[2]

        # Fig creating function for 2d.
        if self.dims == 2:
            def animate(i):
                print(f"Render step {i}/{self.num_steps}")

                fig.clear()
                for j, datacase in enumerate(trajectory_datacases):
                    # Select ax to plot at set boundary.
                    axes[j] = fig.add_subplot(1, 2, j + 1, autoscale_on=False)
                    axes[j].set_aspect("equal")
                    axes[j].set_xlim([float(xboundary[0]), float(xboundary[1])])
                    axes[j].set_ylim([float(yboundary[0]), float(yboundary[1])])
                    axes[j].scatter(self.trajectory[datacase[0]][i][:, 0],
                                        self.trajectory[datacase[0]][i][:, 1], s=point_size, color='blue')
                    axes[j].grid(True, which='both')
                    axes[j].set_title(render_datacases[j])
        '''
        # Fig creating function for 3d
        elif self.dims == 3:
            def animate(i):
                print(f"Render step {i}/{self.num_steps} for {self.output_name}")

                fig.clear()
                for j, datacase in enumerate(trajectory_datacases):
                    # select ax to plot at set boundary
                    axes[j] = fig.add_subplot(1, 2, j + 1, projection='3d', autoscale_on=False)
                    if change_yz == False:
                        axes[j].set_xlim([float(xboundary[0]), float(xboundary[1])])
                        axes[j].set_ylim([float(yboundary[0]), float(yboundary[1])])
                        axes[j].set_zlim([float(zboundary[0]), float(zboundary[1])])
                        for mask, color in color_mask:ModuleNotFoundError: No module named 'pyevtk'
                            axes[j].scatter(self.trajectory[datacase][i][mask, 0],
                                            self.trajectory[datacase][i][mask, 1],
                                            self.trajectory[datacase][i][mask, 2], s=point_size, color=color)
                        # rotate viewpoints angle little by little for each timestep
                        axes[j].set_box_aspect(
                            aspect=(float(xboundary[1]) - float(xboundary[0]),
                                    float(yboundary[1]) - float(yboundary[0]),
                                    float(zboundary[1]) - float(zboundary[0])))
                        axes[j].view_init(elev=vertical_camera_angle, azim=i * viewpoint_rotation)
                        axes[j].grid(True, which='both')
                        axes[j].set_title(render_datacases[j])
                    else:
                        axes[j].set_xlim([float(xboundary[0]), float(xboundary[1])])
                        axes[j].set_ylim([float(zboundary[0]), float(zboundary[1])])
                        axes[j].set_zlim([float(yboundary[0]), float(yboundary[1])])
                        for mask, color in color_mask:
                            axes[j].scatter(self.trajectory[datacase][i][mask, 0],
                                            self.trajectory[datacase][i][mask, 2],
                                            self.trajectory[datacase][i][mask, 1], s=point_size, color=color)
                        # set aspect ratio to equal
                        axes[j].set_box_aspect(
                            aspect=(float(xboundary[1]) - float(xboundary[0]),
                                    float(zboundary[1]) - float(zboundary[0]),
                                    float(yboundary[1]) - float(yboundary[0])))
                        # rotate viewpoints angle little by little for each timestep
                        axes[j].view_init(elev=vertical_camera_angle, azim=i * viewpoint_rotation)
                        axes[j].grid(True, which='both')
                        axes[j].set_title(render_datacases[j])
        '''

        # Create and save animation.
        ani = animation.FuncAnimation(
            fig, animate, frames=np.arange(0, self.num_steps, timestep_stride), interval=10)

        ani.save(f'{self.output_dir}{self.output_name}.gif', dpi=100, fps=30, writer='imagemagick')
        print(f"Animation saved to: {self.output_dir}{self.output_name}.gif")

    def write_vtk(self):
        """
        Write `.vtk` files for each timestep for each rollout case.
        """
        for output_case, label in self.output_cases:
            path = f"{self.output_dir}{self.output_name}_vtk-{label}"
            if not os.path.exists(path):
                os.makedirs(path)
            for i, coord in enumerate(self.trajectory[output_case[0]]):
                pointsToVTK(f"{path}/points{i}",
                            np.array(coord[:, 0]),
                            np.array(coord[:, 1]),
                            np.zeros_like(coord[:, 1]) if self.dims == 2 else np.array(coord[:, 2]),
                            data={"Velocity x": self.velocity_x[output_case[1]][i],
                                  "Velocity y": self.velocity_y[output_case[1]][i]})
                                  #"Velocity magnitude": .sqrt(self.velocity_x[output_case[1]][i]**2 + self.velocity_y[output_case[1]][i]**2)})
        print(f"vtk saved to: {self.output_dir}{self.output_name}...")

    def write_txt(self):
        """
        Write `.txt` files for each timestep for each rollout case.
        """
        for output_case, label in self.output_cases:
            path = f"{self.output_dir}{self.output_name}_txt-{label}"
            if not os.path.exists(path):
                os.makedirs(path)
            for i, coord in enumerate(self.trajectory[output_case[0]]):
                filename = f"{path}/points{i}.txt"
                with open(filename, 'a') as file:
                    file.write("POINTS\n")
                    np.savetxt(file, coord, delimiter='\t', fmt='%.6f')
                with open(filename, 'a') as file:
                    file.write("VELOCITY\n")
                    np.savetxt(file, self.velocity_x[output_case[1]][i], delimiter='\t', fmt='%.6f')
        print(f"txt saved to: {self.output_dir}{self.output_name}...")

def main(_):
    if not FLAGS.output_path:
        raise ValueError("An output directory must be passed with --output_path.")
    if not FLAGS.output_file:
        raise ValueError("A output file name must be passed with --.")

    render = Render(input_dir=FLAGS.output_path, input_name=FLAGS.output_file)

    if FLAGS.output_mode == "gif":
        render.render_gif_animation(
            point_size=1,
            timestep_stride=FLAGS.step_stride,
            vertical_camera_angle=20,
            viewpoint_rotation=0.3,
            change_yz=FLAGS.change_yz
        )
    elif FLAGS.output_mode == "vtk":
        #render.write_vtk()
        render.write_txt()


if __name__ == '__main__':
    app.run(main)