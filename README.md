# C-Uniform Trajectory Sampling

This repository contains the source code for three research papers. 

## Papers

- [1] **C-Uniform Trajectory Sampling For Fast Motion Planning** [(arXiv)](https://arxiv.org/abs/2409.12266) [(website)](https://ogpoyrazoglu.github.io/cuniform_sampling/)
- [2] **An Unsupervised C-Uniform Trajectory Sampler with Applications to Model Predictive Path Integral Control** [(arXiv)](https://arxiv.org/abs/2503.05819)
[(website)](https://rahulmoorthy19.github.io/cu_mppi/)
- [3] **C-Free-Uniform: A Map-Conditioned Trajectory Sampler for Model Predictive Path Integral Control** [(arXiv)](https://arxiv.org/abs/2510.16905) [(website)](https://yukang-cao.github.io/C-Free-Uniform/)

## Overview

This repository is organized into five main directories that each serve a specific purpose in our research.

 - **cuniform_trajectory_sampling**: Contains implementations and scripts related to our trajectory sampling method in [1].
  - **unsupervised_cuniform_trajectory_sampling**: Contains implementations and scripts related to our trajectory sampling method in [2].
- **c_free_uniform_sampling**: Contains implementations and scripts related to our map-conditioned trajectory sampling method in [3].
- **real_world_ros_implementations**: Provides ROS2 implementations for deploying our algorithms in real-world scenarios.

- **simulations**: For end-to-end, closed-loop navigation simulations, see the `navigation_experiments/` module in the **c_free_uniform_sampling** directory ([3]), which provides a self-contained sensor → control → dynamics loop with MPPI, CU-MPPI, and map-conditioned C-Uniform controllers.

For detailed information about each folder, please look at readme files in folders.





