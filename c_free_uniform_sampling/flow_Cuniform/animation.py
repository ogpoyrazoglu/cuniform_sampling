import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.animation import FuncAnimation

def animate_particles(trajectories, filename, num_traj=250, approach='Placeholder'):
    """
    Animate the given trajectories and save the animation to an mp4 file.

    Parameters:
    - trajectories (list): List of trajectories, each trajectory is a list of [x, y, theta].
    - filename (str): Filename to save the animation as an mp4 file.
    - num_traj (int): Number of trajectories (default is 250).
    """
    print("Animating particles...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set the limits for the x and y axes
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    # ax.set_title(f'Network Flow C-Uniform {num_traj} Sampled Trajectories')
    # ax.set_title(f'Neural Network C-Uniform {num_traj} Sampled Trajectories')
    ax.set_title(f'{approach} {num_traj} Sampled Trajectories')
    ax.axis("equal")
    ax.set_xlim(-4, 8)  # X-axis limits
    ax.set_ylim(-4, 4)  # Y-axis limits
    ax.grid(False)

    # Initialize scatter plot for particles at t=0
    scat = ax.scatter([0]*num_traj, [0]*num_traj, c='red', s=30)

    # Initialize quiver (arrows) for each particle
    arrows = ax.quiver([0]*num_traj, [0]*num_traj, [1]*num_traj, [0]*num_traj, angles='xy', scale_units='xy', scale=3, color='darkred', width=0.001) 

    # Store lines for trajectories in the background (light blue)
    lines = []
    for i in range(num_traj):
        line, = ax.plot([], [], color='lightblue', linewidth=0.9, alpha=0.7)
        lines.append(line)

    # Function to initialize the scatter plot and arrows at t=0
    def init():
        scat.set_offsets(np.zeros((num_traj, 2)))  # All particles start at (0, 0)
        arrows.set_UVC([0]*num_traj, [0]*num_traj)
        for line in lines:
            line.set_data([], [])
        return scat, arrows

    # Function to interpolate positions between two frames for smooth motion
    def interpolate(t1, t2, alpha):
        return t1 + alpha * (t2 - t1)

    # Animation function to update particles at each time step
    def update(frame):
        t_fraction = (frame % 5) / 5  # Smooth interpolation (between 0 and 1)
        t_index = frame // 5  # Current time step index
        next_t_index = t_index + 1 if t_index + 1 < len(trajectories[0]) else t_index  # Stay at the last index when finished

        # Interpolate between current and next positions for smooth motion
        x_coords = []
        y_coords = []
        thetas = []

        for i in range(num_traj):
            try:
                if t_index >= len(trajectories[i]) - 1:
                    # This is the last frame, don't interpolate, just take the last state
                    final_state = np.array(trajectories[i][-1])
                    x_coords.append(final_state[0])
                    y_coords.append(final_state[1])
                    thetas.append(final_state[2])
                else:
                    # Handle normal interpolation for intermediate frames
                    current_state = np.array(trajectories[i][t_index])
                    next_state = np.array(trajectories[i][next_t_index])

                    # Interpolating x, y values
                    x_coords.append(interpolate(current_state[0], next_state[0], t_fraction))
                    y_coords.append(interpolate(current_state[1], next_state[1], t_fraction))

                    # Extracting theta (heading angle)
                    thetas.append(current_state[2])

            except IndexError:
                # If the state is incomplete, skip it
                x_coords.append(0)
                y_coords.append(0)
                thetas.append(0)

        # Update particle positions for scatter plot
        positions = np.array(list(zip(x_coords, y_coords)))
        scat.set_offsets(positions)

        # Update arrows based on heading (theta)
        dx = np.cos(thetas)  # Arrow x component
        dy = np.sin(thetas)  # Arrow y component
        arrows.set_offsets(positions)
        arrows.set_UVC(dx, dy)

        # Update the trajectories (light blue) by adding the current waypoint
        for i in range(num_traj):
            # Get previous data and append current positions to the trajectories
            if t_index < len(trajectories[i]) - 1:  # Only update if we're not at the final time step
                x_data, y_data = lines[i].get_data()
                x_data = np.append(x_data, x_coords[i])
                y_data = np.append(y_data, y_coords[i])
                lines[i].set_data(x_data, y_data)

        return scat, arrows, *lines

    # Number of frames is based on the length of the longest trajectory multiplied for smoother transitions
    num_frames = (len(trajectories[0]) - 1) * 5

    # Create animation with slower speed 
    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, repeat=False, interval=100)
    plt.show()
    
    print(f"Saving animation to {filename}...")
    ani.save(filename, writer='ffmpeg', dpi=300)
    print("Saved!")

def main():
    # pickle_file = 'representative_C_Uniform_1000.pickle'
    # pickle_file = 'uniform_sampled_actions_trajectories_1000.pickle'
    # pickle_file = 'nn_interpolation_C_uniform_1000.pickle'
    pickle_file = 'nn_extrapolation_C_Uniform_1000.pickle'
    num_traj = 1000
    output_filename = pickle_file.replace('.pickle', '_animation.mp4')

    with open(pickle_file, 'rb') as file:
        raw_trajectories = pickle.load(file)
        trajectories = [[state_action[0] for state_action in trajectory] for trajectory in raw_trajectories]
        animate_particles(trajectories, output_filename, num_traj, approach="nn_extrapolation")

if __name__ == "__main__":
    main()
