# CU-MPPI ROS2 Codes
This repository contains controller code associated with our submitted paper:

**"An Unsupervised C-Uniform Trajectory Sampler with Applications to Model Predictive Path Integral Control"**

It contains ROS2 codes for real-world F1TENTH navigation. The project was developed on Ubuntu 20.04/ROS2 Foxy, intended for reference. Not plug-and-play solution as it rely on several external nodes, so additional integration work to your setup is needed.

---

## Overview

The repository provides two main types of sampling based controllers that use GPU acceleration via PyCUDA and Numba:

- **CU-MPPI Controller**  
  Implemented in `cumppi_ros2_kinematic.py`. Used C-Uniform trajectory to explore the entire trajectory space and use the minimum cost trajectory as initialization for MPPI's exploitation.

- **MPPI Controller**  
  Implemented in `mppi_ros2_kinematic.py`. 

Additional files include configuration parameters (e.g., `config/nav2_params.yaml` for localization and costmap settings), utility functions, and a setup script for package installation.


---

## Dependencies

Key dependencies include:

- **ROS2 Foxy** on Ubuntu 20.04  
  (Nav2 for localization, AMCL, costmap generation, and other ROS2 node functionalities)
  Checkout [https://github.com/ros-navigation/navigation2] for Nav2 Github page.
  [https://docs.nav2.org/configuration/packages/configuring-amcl.html] for amcl localization.


- **F1Tenth_system**
  Checkout [https://github.com/f1tenth/f1tenth_system]

- **PyCUDA** and **Numba**  
  For GPU acceleration

- Other Python packages such as `numpy`, `scipy`, and standard ROS2 message packages

Make sure your system is correctly set up with these dependencies before attempting to run the controllers.

---

## Running the System

The system is designed to work as part of a larger experimental setup. A typical workflow involves the following steps:

- **Map Generation**
Use SLAM to create a map of the experiment environment. Save the generated map files in the maps folder.

- **Launch Low-Level Control Nodes**
Launch all necessary nodes for F1TENTH low-level control (e.g., joystick, VESC drivers, etc.). 

- **Launch Nav2 Nodes**
Launch all nav2 nodes for AMCL-based localization and costmap generation. An example configuration file is provided in config/nav2_params.yaml.

- **Run Controller Nodes**
Launch the desired controller node by selecting either the cumppi_costmap_node or mppi_costmap_node (as defined in the package’s entry points).
Example command:
```bash
ros2 run f1tenth_controllers cumppi_costmap_node
```

- **Pose Collection**
Run the pose_collector_node to collect trajectory data during experiments.

- **Experiment Execution**
Execute experiments by repeating the above steps

- **Trajectory Analysis**
Use the script at resource/traj_visualization_script.py to visualize and analyze collected trajectories.


---

## Acknowledgements

If you use this code in your research, please consider citing our paper:

@article{poyrazoglu2025unsupervised,
  title={An Unsupervised C-Uniform Trajectory Sampler with Applications to Model Predictive Path Integral Control},
  author={Poyrazoglu, O Goktug and Moorthy, Rahul and Cao, Yukang and Chastek, William and Isler, Volkan},
  journal={arXiv preprint arXiv:2503.05819},
  year={2025}
}

For any questions or suggestions, please contact the maintainer at `cao00125@umn.edu`.
