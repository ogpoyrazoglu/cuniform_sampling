# NN-CUniform

This repository is the implementation and models of NN-CUniform which is part of the paper "An Unsupervised C-Uniform Trajectory Sampler with Applications to Model Predictive Path Integral Control". The steps for running are as follows-

## Model Training

For training the model of NN-CUniform for F1tenth. Please follow the following steps-

1. Get the Level sets Pickle file using the c_uniform_sampling.py code from cuniform_trajectory_sampling folder or download it from [level sets](https://drive.google.com/drive/folders/1KnRhqOXw7AnvsciyZiXCbPs3dhcaYhow?usp=sharing)
2. cd Dubins_Car_Uniformity
3. Change the path of pickle file to your path
4. Change the save folder path
5. Run the following command-
```
python Neural_C_Uniform.py
```

## Trajectory Sampling

For sampling the trajectories, follow the following steps

1. Get the Level sets Pickle file using the c_uniform_sampling.py code from cuniform_trajectory_sampling folder or download it from [level sets](https://drive.google.com/drive/folders/1KnRhqOXw7AnvsciyZiXCbPs3dhcaYhow?usp=sharing)
2. Change the model path to your path
3. Change the time horizon, discretization and number of trajectories to your requirement
4. Run the following command
```
python Trajectory_sampling.py
```
We also have a graph sampling version to have the exact same trajectory sampling as our base paper "C-Uniform Trajectory Sampling For Fast Motion Planning". For having that version, please run the following command-
```
python NN_trajectory_graph_creation.py
python NN_graph_trajectory_sampling.py
```
## Uniformity Analysis

For understanding the uniformity % of your model, please follow the following steps

1. Get the Level sets Pickle file using the c_uniform_sampling.py code from cuniform_trajectory_sampling folder or download it from [level sets](https://drive.google.com/drive/folders/1KnRhqOXw7AnvsciyZiXCbPs3dhcaYhow?usp=sharing)
2. Change the model path to your path
3. Run the following command
```
python Uniformity_analysis.py
```

## Trajectory visualization and Reachable visualization

For getting the visualization, similar to the paper. Please follow the following steps

1. Get the Level sets Pickle file using the c_uniform_sampling.py code from cuniform_trajectory_sampling folder or download it from [level sets](https://drive.google.com/drive/folders/1KnRhqOXw7AnvsciyZiXCbPs3dhcaYhow?usp=sharing)
2. Change the model path to your path
3. Change the trajectories path to your path
3. Run the following command
```
python Trajectory_visualization.py or python Trajectory_reachable_area_visualization.py
```

## 1D Random Walk

These are the initial experiment folder ran on 1D random walk environment to understand the working of the loss function.
