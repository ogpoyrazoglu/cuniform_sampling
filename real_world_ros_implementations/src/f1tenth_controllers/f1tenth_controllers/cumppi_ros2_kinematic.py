#!/usr/bin/env python3
# ROS 2 packages
import rclpy
from tf2_ros import Buffer, TransformListener # for locolization
import traceback
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker
from geometry_msgs.msg import TransformStamped  # Use TransformStamped instead of Rigids
from collections import deque  # For implementing a circular buffer
from rclpy.qos import qos_profile_sensor_data
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import OccupancyGrid # for applying local costmap
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import std_msgs

import numpy as np
import math
import copy
import numba
import time
import sys 
import pickle

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Import pyCuda modules for computations
from pycuda.curandom import XORWOWRandomNumberGenerator
from pycuda import gpuarray

# Information about your GPU
# Initialize PyCUDA and create a primary context
cuda.init()
device = cuda.Device(0)
primary_context = device.retain_primary_context()
primary_context.push()
# primary_context.pop()

from numba import cuda as numba_cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
from .cuda_device_functions import *
from .input_constraints import *
# import utils.mppi_nln_utils as uls 
gpu = numba_cuda.get_current_device()
print(numba_cuda.is_available())
max_threads_per_block = gpu.MAX_THREADS_PER_BLOCK
max_square_block_dim = (int(gpu.MAX_BLOCK_DIM_X**0.5), int(gpu.MAX_BLOCK_DIM_X**0.5))
max_blocks = gpu.MAX_GRID_DIM_X
max_rec_blocks = rec_max_control_rollouts = int(1e6) # Though theoretically limited by max_blocks on GPU
rec_min_control_rollouts = 100
np.set_printoptions(precision=2, suppress=True)

ACTION_WEIGHT = 1.0
num_traj = 1000 # only use the first 'num_traj' trajectories
print("Reading cuniform trajectories...")
with open("/home/nvidia/f1tenth_ws/src/f1tenth_controllers/resource/......", 'rb') as f:
    #NOTE: change this part to load the cuniform trajectories
    cuniform_trajectories = pickle.load(f)[:num_traj]

# Function to handle None in the scalar part
def process_element(array_part, scalar_part):
    # Replace None scalar with 0.0
    if scalar_part is None:
        scalar_part = 0.0
    return np.concatenate([array_part, [scalar_part]])

# Process all trajectories
print("Processing cuniform trajectories...")
processed_trajectories = [
    np.array([process_element(array_part, scalar_part) for array_part, scalar_part in trajectory])
    for trajectory in cuniform_trajectories
]

# Combine into a 3D array if all trajectories are of the same size
cuniform_trajectories_transformed = np.array(processed_trajectories)

import os
class Config:
  """ Configurations that are typically fixed throughout execution. """
  def __init__(self, 
               T=3, # Horizon (s)
               dt=0.2, # Length of each step (s)
               num_control_rollouts=1000, # Number of control sequences
               num_vis_state_rollouts=1000, # Number of visualization rollouts
               seed=1):
    self.seed = seed
    self.T = T
    self.dt = dt
    self.num_steps = int(T/dt)
    self.max_threads_per_block = max_threads_per_block # save just in case

    assert T > 0
    assert dt > 0
    assert T > dt
    assert self.num_steps > 0
    
    # Number of control rollouts are currently limited by the number of blocks
    self.num_control_rollouts = num_control_rollouts
    if self.num_control_rollouts > rec_max_control_rollouts:
      self.num_control_rollouts = rec_max_control_rollouts
      print("MPPI Config: Clip num_control_rollouts to be recommended max number of {}. (Max={})".format(
        rec_max_control_rollouts, max_blocks))
    elif self.num_control_rollouts < rec_min_control_rollouts:
      self.num_control_rollouts = rec_min_control_rollouts
      print("MPPI Config: Clip num_control_rollouts to be recommended min number of {}. (Recommended max={})".format(
        rec_min_control_rollouts, rec_max_control_rollouts))
    
    # For visualizing state rollouts
    self.num_vis_state_rollouts = num_vis_state_rollouts
    self.num_vis_state_rollouts = min([self.num_vis_state_rollouts, self.num_control_rollouts])
    self.num_vis_state_rollouts = max([1, self.num_vis_state_rollouts])
DEFAULT_OBS_COST = 1e4
DEFAULT_DIST_WEIGHT = 10

class CUniform_Numba(object):
  """ 
  Planner object that initializes GPU memory and runs MPPI on GPU via numba.   
  CURRENT IMPLEMENATION will one calculate the cost of the each trajectory and select the one with the minimum one. 
  """
  def __init__(self, cfg):
    # Fixed configs
    self.cfg = cfg
    self.T = cfg.T
    self.dt = cfg.dt
    self.num_steps = cfg.num_steps
    self.num_control_rollouts = cfg.num_control_rollouts

    self.num_vis_state_rollouts = cfg.num_vis_state_rollouts
    self.seed = cfg.seed
    self.vehicle_length = 0.57
    self.vehicle_width = 0.3
    self.vehicle_wheelbase = 0.32
    self.x0 = None
    # Basic info 
    self.max_threads_per_block = cfg.max_threads_per_block

    # Initialize reuseable device variables
    self.u_cur_d = None
    self.u_prev_d = None
    self.costs_d = None
    self.feasible_mask_d = None           # This variable and one below is new
    self.num_feasible_d = np.float32(0.0)
    self.weights_d = None
    self.rng_states_d = None
    self.state_rollout_batch_d = None # For visualization only. Otherwise, inefficient

    # Other task specific params
    self.device_var_initialized = False
    self.generator = XORWOWRandomNumberGenerator()

    # Task specific params
    # local costmap size and resolution
    self.local_costmap_size = 120
    self.local_costmap_resolution = 0.05
    self.reset()
    
  def reset(self):
    # Other task specific params
    self.u_seq0 = np.zeros((self.num_steps, 2), dtype=np.float32)
    self.u_seq0[:,0] = 1.0 # Linear velocity
    self.params = None
    self.params_set = False
    self.costmap_loaded = False
    self.u_prev_d = None
    
    # Initialize all fixed-size device variables ahead of time. (Do not change in the lifetime of MPPI object)
    self.init_device_vars_before_solving()

  def load_trajectories(self, trajectories):
    self.trajectories = copy.deepcopy(trajectories)
    self.trajectories_d = numba_cuda.to_device(self.trajectories.astype(np.float32))
    self.original_trajectories_d = numba_cuda.to_device(self.trajectories.astype(np.float32))

  def init_device_vars_before_solving(self):
    if not self.device_var_initialized:
      t0 = time.time()
      # Useq
      self.u_cur_d = numba_cuda.to_device(self.u_seq0) 
      self.u_prev_d = numba_cuda.to_device(self.u_seq0)

      self.costs_d = numba_cuda.device_array((self.num_control_rollouts), dtype=np.float32)
      self.feasible_mask_d = numba_cuda.device_array((self.num_control_rollouts, self.num_steps+1), dtype=np.float32)
      # add full ones to the feasible mask
      self.feasible_mask_d[0:self.num_control_rollouts, 0:self.num_steps+1] = 1.0
      self.weights_d = numba_cuda.device_array((self.num_control_rollouts), dtype=np.float32)
      self.local_costmap_d = numba_cuda.device_array((self.local_costmap_size, self.local_costmap_size), dtype=np.float32)  
       
      self.device_var_initialized = True
      print(" CUniform planner has initialized GPU memory after {} s".format(time.time()-t0))

  def setup(self, params):
    # These tend to change (e.g., current robot position, the map) after each step
    self.set_params(params)

  def set_params(self, params):
    self.params = copy.deepcopy(params)
    self.x0 = self.params['x0']
    if self.params['costmap'] is not None: # should be type ndarray
        self.costmap_loaded = True
    self.params_set = True

  def check_solve_conditions(self):
    if not self.params_set:
      print("MPPI parameters are not set. Cannot solve")
      return False
    if not self.device_var_initialized:
      print("Device variables not initialized. Cannot solve.")
      return False
    if not self.costmap_loaded:
      print("Costmap not loaded. Cannot solve.")
      return False
    return True

  def solve(self):
    """Entry point for different algoritims"""
    if not self.check_solve_conditions():
      print("C-Uniform solve condition not met. Cannot solve. Return")
      return
    return self.get_rollout_cost()
  
  def move_cuniform_task_vars_to_device(self):
    xgoal_d = numba_cuda.to_device(self.params['xgoal'].astype(np.float32))
    x0_d = numba_cuda.to_device(self.params['x0'].astype(np.float32))
    goal_tolerance_d = np.float32(self.params['goal_tolerance'])

    vehicle_length_d = np.float32(self.vehicle_length)
    vehicle_width_d = np.float32(self.vehicle_width)
    vehicle_wheelbase_d = np.float32(self.vehicle_wheelbase)

    #USEQ UPDATE
    lambda_weight_d = np.float32(self.params['lambda_weight'])
    vrange_d = numba_cuda.to_device(self.params['vrange'].astype(np.float32))

    ''' COSTMAP Variables'''
    local_costmap_edt = self.params['costmap']
    local_costmap_edt = np.ascontiguousarray(local_costmap_edt)
    max_local_cost_d = np.float32(np.max(local_costmap_edt))
    # set the local costmap to the local_costmap_d on the device
    local_costmap_d = numba_cuda.to_device(local_costmap_edt)
    costmap_origin_x = np.float32(self.params['costmap_origin'][0])
    costmap_origin_y = np.float32(self.params['costmap_origin'][1])
    costmap_resolution = np.float32(self.params['costmap_resolution'])

    #obstacle
    obs_cost_d = np.float32(DEFAULT_OBS_COST if 'obs_penalty' not in self.params 
                                  else self.params['obs_penalty'])

    return xgoal_d, x0_d, goal_tolerance_d, \
            vehicle_length_d, vehicle_width_d, vehicle_wheelbase_d, \
            lambda_weight_d, vrange_d, \
            obs_cost_d, local_costmap_d, max_local_cost_d, \
            costmap_origin_x, costmap_origin_y, costmap_resolution

  def get_rollout_cost(self):
    '''
    Calculate the cost of the each trajectories and find the trajectory with min cost. and return that trajectory to the host
    '''
    xgoal_d, x0_d, goal_tolerance_d, \
    vehicle_length_d, vehicle_width_d, vehicle_wheelbase_d, \
    lambda_weight_d, vrange_d, \
    obs_cost_d, local_costmap_d, max_local_cost_d, \
    costmap_origin_x, costmap_origin_y, costmap_resolution, \
    = self.move_cuniform_task_vars_to_device()

    # Weight for distance cost
    dist_weight = DEFAULT_DIST_WEIGHT if 'dist_weight' not in self.params else self.params['dist_weight']

    self.rollouts_cost_numba[self.num_control_rollouts, 1](
      self.trajectories_d,
      self.costs_d,
      goal_tolerance_d,
      xgoal_d,
      x0_d,
      obs_cost_d,
      local_costmap_d,
      costmap_origin_x,
      costmap_origin_y,
      costmap_resolution,
      max_local_cost_d,
      vehicle_length_d,
      vehicle_width_d,
      vehicle_wheelbase_d,
      dist_weight,
    )

    # # get the cost of the trajectories that are feasible to the host
    cost_arr = self.costs_d.copy_to_host()
    min_cost_index = np.argmin(cost_arr)

    self.u_prev_d = self.u_cur_d
    return min_cost_index, self.costs_d.copy_to_host(), self.trajectories_d[min_cost_index].copy_to_host(), self.u_cur_d.copy_to_host()

  def get_state_rollout(self, x_curr, trajectories):
    # First translate the point
    transformed_trajectories = copy.deepcopy(trajectories) #RUNTIME: negeligible

    # Then rotate the point, Rotation matrix
    theta = x_curr[2]
    translation = x_curr[:2]
    R = np.array([[math.cos(x_curr[2]), -math.sin(x_curr[2])], [math.sin(x_curr[2]), math.cos(x_curr[2])]])
    transformed_trajectories[:, :, :2] = np.einsum('ij,nmj->nmi', R, transformed_trajectories[:, :, :2])

    transformed_trajectories[:, :, 2] += theta
    transformed_trajectories[:, :, :2] += translation
    return transformed_trajectories
    
  def shift_and_update(self, x_next, trajectories):
    trajs = copy.deepcopy(trajectories)
    transformed_trajs = self.get_state_rollout(x_next, trajs)
    self.load_trajectories(transformed_trajs)
    self.x0 = x_next.copy()

  """GPU kernels from here on"""
  @staticmethod
  @numba_cuda.jit(fastmath=True)
  def rollouts_cost_numba(
    trajectories_d,
    costs_d,
    goal_tolerance_d,
    xgoal_d,
    x0_d,
    obs_cost_d,
    local_costmap_d,
    costmap_origin_x,
    costmap_origin_y,
    params_costmap_resolution,
    max_local_cost_d,
    vehicle_length_d,
    vehicle_width_d,
    vehicle_wheelbase_d,
    dist_weight_d,
  ):
    """
    There should only be one thread running in each block, where each block handles a single sampled trajecotry calulation.
    """
    # Get block id and thread id
    bid = numba_cuda.blockIdx.x   # index of block
    tid = numba_cuda.threadIdx.x  # index of thread within a block

    # Initialize the cost for the trajectory
    costs_d[bid] = 0.0
    goal_reached = False
    isCollided = False
    goal_tolerance_d2 = goal_tolerance_d*goal_tolerance_d
    dist_to_goal2 = prev_dist_to_goal2 = 1e9 # initialize to a large value

    # Allocate space for vehicle boundary points (4)
    x_curr = numba_cuda.local.array(3, numba.float32) # Dubins car model states x,y,theta
    x_curr_grid_d = numba_cuda.local.array((2), dtype=np.int32)
    
    gamma = 1.0 # Discount factor for cost
    # Loop through each state in the trajectory
    num_steps = trajectories_d.shape[1]
    for step in range(num_steps):
        # Extract current state (x, y, theta)
        for i in range(3):
          x_curr[i] = trajectories_d[bid, step, i]
        # steering_angle = trajectories_d[bid, step, 3]
        convert_position_to_costmap_indices_gpu(
            x_curr[0],
            x_curr[1],
            costmap_origin_x,
            costmap_origin_y,
            params_costmap_resolution,
            x_curr_grid_d,
        )
        # obstacle cost
        costs_d[bid] += calculate_localcostmap_cost(local_costmap_d, x_curr_grid_d) / (100) * obs_cost_d

        if not isCollided:
          # Check for collision
          if check_state_collision_gpu(local_costmap_d, x_curr_grid_d) == 1.0:
            isCollided = True

          # action cost
          # costs_d[bid] += ACTION_WEIGHT * math.fabs(steering_angle)
            
          # distance to goal cost
          dist_to_goal2 = (((xgoal_d[0]-x_curr[0])**2 + (xgoal_d[1]-x_curr[1])**2))**0.5
          costs_d[bid] += stage_cost(dist_to_goal2, 5.0)
          
          if dist_to_goal2  <= goal_tolerance_d:
            goal_reached = True
            break
          prev_dist_to_goal2 = dist_to_goal2
        else:
          # costs_d[bid] += 1 * obs_cost_d
          # stage cost
          costs_d[bid] += stage_cost(prev_dist_to_goal2, 5.0)

    # Accumulate terminal cost 
    costs_d[bid] += term_cost(dist_to_goal2, goal_reached)
    # give reward for reaching the goal
    # costs_d[bid] += (-goal_reached) * 10

class MPPI_Numba(object):
  """ 
  Implementation of Information theoretic MPPI by Williams et. al. 
  Alg 2. in https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf

  Planner object that initializes GPU memory and runs MPPI on GPU via numba. 
  
  Typical workflow: 
    1. Initialize object with config that allows pre-initialization of GPU memory
    2. reset()
    3. setup(mppi_params) based on problem instance
    4. solve(), which returns optimized control sequence
    5. get_state_rollout() for visualization
    6. shift_and_update(next_state, optimal_u_sequence, num_shifts=1)
    7. Repeat from 2 if params have changed
  """
  def __init__(self, cfg):
    # Fixed configs
    self.cfg = cfg
    self.T = cfg.T
    self.dt = cfg.dt
    self.num_steps = cfg.num_steps
    self.num_control_rollouts = cfg.num_control_rollouts

    self.num_vis_state_rollouts = cfg.num_vis_state_rollouts
    self.seed = cfg.seed
    self.vehicle_length = 0.57
    self.vehicle_width = 0.3
    self.vehicle_wheelbase = 0.32
    # Basic info 
    self.max_threads_per_block = cfg.max_threads_per_block

    # Initialize reuseable device variables
    self.noise_samples_d = None
    self.u_cur_d = None
    self.u_prev_d = None
    self.costs_d = None
    self.weights_d = None
    self.rng_states_d = None
    self.state_rollout_batch_d = None # For visualization only. Otherwise, inefficient

    # Other task specific params
    self.device_var_initialized = False

    self.generator = XORWOWRandomNumberGenerator()
    ''' ################ IMPORTANT ################
        # NOTE: adjust mppi type here
        ################ IMPORTANT ################ ''' 
    self.mppi_type = 1 # Normal dist / 1: NLN

    if self.mppi_type == 1:
      self.mu_LogN, self.std_LogN = Normal2LogN(0, np.mean([0.1, 0.1]))
      self.rLogN_info = [self.mppi_type, self.mu_LogN, self.std_LogN]

    # local costmap size and resolution
    self.local_costmap_size = 120
    self.local_costmap_resolution = 0.05
    self.reset()
    
  def reset(self):
    # Other task specific params
    self.u_seq0 = np.zeros((self.num_steps, 2), dtype=np.float32)
    self.u_seq0[:,0] = 1.0 # Linear speed
    self.params = None
    self.params_set = False
    self.costmap_loaded = False
    self.u_prev_d = None
    
    # Initialize all fixed-size device variables ahead of time. (Do not change in the lifetime of MPPI object)
    self.init_device_vars_before_solving()

  def init_device_vars_before_solving(self):
    if not self.device_var_initialized:
      t0 = time.time()
      self.noise_samples_d = numba_cuda.device_array((self.num_control_rollouts, self.num_steps, 2), dtype=np.float32) # to be sampled collaboratively via GPU
      self.u_cur_d = numba_cuda.to_device(self.u_seq0) 
      self.u_prev_d = numba_cuda.to_device(self.u_seq0) 
      self.costs_d = numba_cuda.device_array((self.num_control_rollouts), dtype=np.float32)
      self.weights_d = numba_cuda.device_array((self.num_control_rollouts), dtype=np.float32)
      self.rng_states_d = create_xoroshiro128p_states(self.num_control_rollouts*self.num_steps, seed=self.seed)
      
      self.state_rollout_batch_d = numba_cuda.device_array((self.num_vis_state_rollouts, self.num_steps+1, 3), dtype=np.float32)   
      self.local_costmap_d = numba_cuda.device_array((self.local_costmap_size, self.local_costmap_size), dtype=np.float32)   
      self.device_var_initialized = True
      print("MPPI planner has initialized GPU memory after {} s".format(time.time()-t0))
    else:
      print('first actions are reinitialized')
      self.u_cur_d = numba_cuda.to_device(self.u_seq0)
      self.u_prev_d = numba_cuda.to_device(self.u_seq0)

  def setup(self, params):
    # These tend to change (e.g., current robot position, the map) after each step
    self.set_params(params)
  
  def set_params(self, params):
    self.params = copy.deepcopy(params)
    self.params_set = True
    if self.params['costmap'] is not None: # should be type ndarray
      self.costmap_loaded = True

  def set_actions(self, u_seq):
    u_seq_dummy = copy.deepcopy(u_seq)
    self.u_cur_d = numba_cuda.to_device(u_seq_dummy.astype(np.float32))

  def check_solve_conditions(self):
    if not self.params_set:
      print("MPPI parameters are not set. Cannot solve")
      return False
    if not self.device_var_initialized:
      print("Device variables not initialized. Cannot solve.")
      return False
    if not self.costmap_loaded:
      print("Costmap not loaded. Cannot solve.")
      return False
    return True

  def solve(self):
    """Entry point for different algoritims"""
    if not self.check_solve_conditions():
      print("MPPI solve condition not met. Cannot solve. Return")
      return
    return self.solve_with_nominal_dynamics()

  def random_noise_sample(self):
    # Use the random generator to generate random noise
    # The logic from log-MPPI_ros github repo
    if self.mppi_type == 0: # Normal Dist
      du_d = self.generator.gen_normal(
          self.num_control_rollouts * self.num_steps * 2,
          np.float32)

    # log-MPPI
    elif self.mppi_type == 1: # NLN
        du_LogN_d = self.generator.gen_log_normal(
            self.num_control_rollouts * self.num_steps * 2,
            np.float32, self.mu_LogN, self.std_LogN)
        du_d = du_LogN_d * self.generator.gen_normal(
            self.num_control_rollouts * self.num_steps * 2,
            np.float32)
    return du_d.get()

  def move_mppi_task_vars_to_device(self):
    vrange_d = numba_cuda.to_device(self.params['vrange'].astype(np.float32))
    wrange_d = numba_cuda.to_device(self.params['wrange'].astype(np.float32))
    xgoal_d = numba_cuda.to_device(self.params['xgoal'].astype(np.float32))
    goal_tolerance_d = np.float32(self.params['goal_tolerance'])
    lambda_weight_d = np.float32(self.params['lambda_weight'])

    u_std_d = numba_cuda.to_device(self.params['u_std'].astype(np.float32))
    self.u_std = self.params['u_std'].astype(np.float32)
    x0_d = numba_cuda.to_device(self.params['x0'].astype(np.float32))
    dt_d = np.float32(self.params['dt'])
    vehicle_length_d = np.float32(self.vehicle_length)
    vehicle_width_d = np.float32(self.vehicle_width)
    vehicle_wheelbase_d = np.float32(self.vehicle_wheelbase)
    v_switch_d = np.float32(self.params['v_switch'])

    local_costmap_edt = self.params['costmap']
    local_costmap_edt = np.ascontiguousarray(local_costmap_edt)
    max_local_cost_d = np.float32(np.max(local_costmap_edt))
    # set the local costmap to the local_costmap_d on the device
    local_costmap_d = numba_cuda.to_device(local_costmap_edt)
    obs_cost_d = np.float32(DEFAULT_OBS_COST if 'obs_penalty' not in self.params 
                                  else self.params['obs_penalty'])
    costmap_origin_x = np.float32(self.params['costmap_origin'][0])
    costmap_origin_y = np.float32(self.params['costmap_origin'][1])
    costmap_resolution = np.float32(self.params['costmap_resolution'])
    return vrange_d, wrange_d, xgoal_d, \
           goal_tolerance_d, lambda_weight_d, \
           vehicle_length_d, vehicle_width_d, vehicle_wheelbase_d, \
           v_switch_d, \
           u_std_d, x0_d, dt_d, local_costmap_d, obs_cost_d, max_local_cost_d, \
           costmap_origin_x, costmap_origin_y, costmap_resolution

  def solve_with_nominal_dynamics(self):
    """
    Launch GPU kernels that use nominal dynamics but adjsuts cost function based on worst-case linear speed.
    """
    vrange_d, wrange_d, xgoal_d, goal_tolerance_d, lambda_weight_d, \
    vehicle_length_d, vehicle_width_d, vehicle_wheelbase_d,\
    v_switch_d, \
    u_std_d, x0_d, dt_d, local_costmap_d, obs_cost_d, max_local_cost_d, \
    costmap_origin_x, costmap_origin_y, params_costmap_resolution = self.move_mppi_task_vars_to_device()

    # Weight for distance cost
    dist_weight = DEFAULT_DIST_WEIGHT if 'dist_weight' not in self.params else self.params['dist_weight']

    # Optimization loop
    for k in range(self.params['num_opt']):
      # Sample control noise
      noise_samples = self.random_noise_sample()
      # reshape the noise samples to (num_control_rollouts, num_steps, 2)
      noise_samples_reshaped = noise_samples.reshape(self.num_control_rollouts, self.num_steps, 2).astype(np.float32)
      noise_samples_reshaped[:,:,0] *= u_std_d[0]
      noise_samples_reshaped[:,:,1] *= u_std_d[1]
      self.noise_samples_d = numba_cuda.to_device(noise_samples_reshaped)

      # Rollout and compute mean or cvar
      self.rollouts_cost_numba[self.num_control_rollouts, 1](
        vrange_d,
        wrange_d,
        xgoal_d,
        local_costmap_d, # local_costmap added
        max_local_cost_d,
        obs_cost_d,
        vehicle_length_d,
        vehicle_width_d,
        vehicle_wheelbase_d,
        goal_tolerance_d,
        lambda_weight_d,
        u_std_d,
        x0_d,
        dt_d,
        dist_weight,
        v_switch_d, 
        self.noise_samples_d,
        self.u_cur_d,
        costmap_origin_x,
        costmap_origin_y,
        params_costmap_resolution,

        # results
        self.costs_d,
      )      
      self.u_prev_d = self.u_cur_d

      # Compute cost and update the optimal control on device
      self.update_useq_numba[1, 32](
        lambda_weight_d, 
        self.costs_d, 
        self.noise_samples_d, 
        self.weights_d, 
        vrange_d,
        wrange_d,
        self.u_cur_d
      )
    return self.u_cur_d.copy_to_host()

  def shift_and_update(self, new_x0, u_cur, num_shifts=1):
    self.params["x0"] = new_x0.copy() #NOTE: mppi_dubins_moving_target_ros_node.py have commentted this line out
    self.shift_optimal_control_sequence(u_cur, num_shifts)

  def shift_optimal_control_sequence(self, u_cur, num_shifts=1):
    u_cur_shifted = u_cur.copy()
    u_cur_shifted[:-num_shifts] = u_cur_shifted[num_shifts:]
    self.u_cur_d = numba_cuda.to_device(u_cur_shifted.astype(np.float32))

  """GPU kernels from here on"""
  @staticmethod
  @numba_cuda.jit(fastmath=True)
  def rollouts_cost_numba(
          vrange_d, 
          wrange_d, 
          xgoal_d, 
          local_costmap_d,
          max_local_cost_d, # local_costmap added
          obs_cost_d,
          vehicle_length_d,
          vehicle_width_d,
          vehicle_wheelbase_d,         
          goal_tolerance_d, 
          lambda_weight_d, 
          u_std_d, 
          x0_d, 
          dt_d,
          dist_weight_d,
          v_switch_d,
          noise_samples_d,
          u_cur_d,
          costmap_origin_x,
          costmap_origin_y,
          params_costmap_resolution,
          costs_d,
        ):
    """
    There should only be one thread running in each block, where each block handles a single sampled control sequence.
    """
    # Get block id and thread id
    bid = numba_cuda.blockIdx.x   # index of block
    costs_d[bid] = 0.0

    # Explicit unicycle update and map lookup
    x_curr = numba_cuda.local.array(3, numba.float32) # x, y, theta
    x_curr_grid_d = numba_cuda.local.array((2), dtype=np.int32)

    for i in range(3): 
      x_curr[i] = x0_d[i]

    timesteps = len(u_cur_d)
    goal_reached = False
    isCollided = False

    # squared goal tolerance
    goal_tolerance_d2 = goal_tolerance_d*goal_tolerance_d
    dist_to_goal2 = prev_dist_to_goal2 = 1e9

    v_nom = v_noisy = w_nom = w_noisy = 0.0

    gamma = 1.0
    for t in range(timesteps):
      # Nominal noisy control
      w_nom = u_cur_d[t, 1] + noise_samples_d[bid, t, 1]
      w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))
      v_noisy = vrange_d[0] # fixed speed 1.0

      # Forward simulate
      # kinematic model update
      x_curr[0] += dt_d*v_noisy*math.cos(x_curr[2])
      x_curr[1] += dt_d*v_noisy*math.sin(x_curr[2])
      x_curr[2] += dt_d*v_noisy*math.tan(w_noisy)/vehicle_wheelbase_d
      # x_curr[2] = math.fmod(x_curr[2], 2*math.pi)

      # obstacle cost
      convert_position_to_costmap_indices_gpu(
        x_curr[0],
        x_curr[1],
        costmap_origin_x,
        costmap_origin_y,
        params_costmap_resolution,
        x_curr_grid_d,
      )
      costs_d[bid] += calculate_localcostmap_cost(local_costmap_d, x_curr_grid_d) / (100) * obs_cost_d

      # Check the state is collided with the obstacle
      # Get current state costmap indices
      if not isCollided:
        # Check for collision
        if check_state_collision_gpu(local_costmap_d, x_curr_grid_d) == 1.0:
          isCollided = True

        # distance to goal cost
        dist_to_goal2 = (((xgoal_d[0]-x_curr[0])**2) + ((xgoal_d[1]-x_curr[1])**2)) ** 0.5
        costs_d[bid] += stage_cost(dist_to_goal2, 5.0)

        if dist_to_goal2 <= goal_tolerance_d:
          goal_reached = True
          break
        prev_dist_to_goal2 = dist_to_goal2
      else:
        costs_d[bid] += stage_cost(prev_dist_to_goal2, 5.0)# distance to goal cost
    # Accumulate terminal cost 
    costs_d[bid] += term_cost(dist_to_goal2, goal_reached)

  @staticmethod
  @numba_cuda.jit(fastmath=True)
  def update_useq_numba(
        lambda_weight_d,
        costs_d,
        noise_samples_d,
        weights_d,
        vrange_d,
        wrange_d,
        u_cur_d):
    """
    GPU kernel that updates the optimal control sequence based on previously evaluated cost values.
    Assume that the function is invoked as update_useq_numba[1, NUM_THREADS], with one block and multiple threads.
    """
    tid = numba_cuda.threadIdx.x
    num_threads = numba_cuda.blockDim.x
    numel = len(noise_samples_d)
    gap = int(math.ceil(numel / num_threads))

    # Find the minimum value via reduction
    starti = min(tid*gap, numel)
    endi = min(starti+gap, numel)
    if starti<numel:
      weights_d[starti] = costs_d[starti]
    for i in range(starti, endi):
      weights_d[starti] = min(weights_d[starti], costs_d[i])
    numba_cuda.syncthreads()

    s = gap
    while s < numel:
      if (starti % (2 * s) == 0) and ((starti + s) < numel):
        # Stride by `s` and add
        weights_d[starti] = min(weights_d[starti], weights_d[starti + s])
      s *= 2
      numba_cuda.syncthreads()

    beta = weights_d[0]
    
    # Compute weight
    for i in range(starti, endi):
      weights_d[i] = math.exp(-1./lambda_weight_d*(costs_d[i]-beta))
    numba_cuda.syncthreads()

    # Normalize
    # Reuse costs_d array
    for i in range(starti, endi):
      costs_d[i] = weights_d[i]
    numba_cuda.syncthreads()
    for i in range(starti+1, endi):
      costs_d[starti] += costs_d[i]
    numba_cuda.syncthreads()
    s = gap
    while s < numel:
      if (starti % (2 * s) == 0) and ((starti + s) < numel):
        # Stride by `s` and add
        costs_d[starti] += costs_d[starti + s]
      s *= 2
      numba_cuda.syncthreads()

    for i in range(starti, endi):
      weights_d[i] /= costs_d[0]
    numba_cuda.syncthreads()
    
    # update the u_cur_d
    timesteps = len(u_cur_d)
    for t in range(timesteps):
      for i in range(starti, endi):
        numba_cuda.atomic.add(u_cur_d, (t, 0), weights_d[i]*noise_samples_d[i, t, 0])
        numba_cuda.atomic.add(u_cur_d, (t, 1), weights_d[i]*noise_samples_d[i, t, 1])
    numba_cuda.syncthreads()

    # Blocks crop the control together
    tgap = int(math.ceil(timesteps / num_threads))
    starti = min(tid*tgap, timesteps)
    endi = min(starti+tgap, timesteps)
    for ti in range(starti, endi):
      u_cur_d[ti, 0] = max(vrange_d[0], min(vrange_d[1], u_cur_d[ti, 0]))
      u_cur_d[ti, 1] = max(wrange_d[0], min(wrange_d[1], u_cur_d[ti, 1]))

class CUMPPIPlannerNode(Node):
    def __init__(self):
        super().__init__('CUMPPI_planner_node')
        self.cfg = Config(
            T = 3,
            dt = 0.2,
            seed = 1,
        )
        self.cuniform_cfg = Config(
            T = 3,
            dt = 0.2,
            num_control_rollouts = 1000, # Same as number of blocks, can be more than 1024
            num_vis_state_rollouts = 1,
            seed = 1,
        )
        self.mppi_cfg = Config(
            T = 3,
            dt = 0.2,
            num_control_rollouts = 500, # Same as number of blocks, can be more than 1024
            num_vis_state_rollouts = 1,
            seed = 1,
        )
        
        # Define initial parameters for the c_uniform planner.
        self.cuniform_params = dict(
          # Task specification
          dt = self.cfg.dt, 
          x0 = np.zeros(3), # Start state
          xgoal = np.array([-1.0, -15.0]), # Goal position
          # vehicle length(lf and lr wrt the cog) and width
          vehicle_length = 0.57,
          vehicle_width = 0.3,
          vehicle_wheelbase= 0.32,
          # For risk-aware min time planning
          goal_tolerance = 0.40,
          dist_weight = 1e2, #  Weight for dist-to-goal cost.
          num_opt = 1, # Number of steps in each solve() function call.
          lambda_weight = 1.0, # Temperature param in MPPI
          # Control and sample specification
          vrange = np.array([1.0, 1.0]), # Linear velocity range. Constant Linear Velocity
          
          costmap = None, # intiallly nothing
          obs_penalty = 1e2
        )
        
        # Instantiate the c_uniform planner and load trajectories
        self.cuniform = CUniform_Numba(self.cuniform_cfg)
        self.original_trajectories = cuniform_trajectories_transformed
        self.cuniform.setup(self.cuniform_params)
        self.cuniform.load_trajectories(self.original_trajectories[:])
        self.get_logger().info('Cuniform NUMBA initialized.')

        # Define initial parameters for the MPPI planner
        self.mppi_params = dict(
          # Task specification
          dt = self.cfg.dt, 
          x0 = np.zeros(3), # Start state
          xgoal = np.array([-1.0, -15.0]), # Goal position
          # vehicle length(lf and lr wrt the cog) and width
          vehicle_length = 0.57,
          vehicle_width = 0.3,
          vehicle_wheelbase= 0.32,
          # For risk-aware min time planning
          goal_tolerance = 0.40,
          dist_weight = 10, #  Weight for dist-to-goal cost.

          lambda_weight = 1.0, # Temperature param in MPPI
          num_opt = 1, # Number of steps in each solve() function call.

          # Control and sample specification
          u_std = np.array([0.023, 0.1]), # Noise std for sampling linear and angular velocities.
          v_switch = 0.2, 
          vrange = np.array([1.0, 1.0]), # Linear velocity range. Constant Linear Velocity
          wrange = np.array([-np.pi/6, np.pi/6]), # Angular velocity range.
          costmap = None, # intiallly nothing
          obs_penalty = 1e2
        )
        # Instantiate the MPPI planner
        self.mppi = MPPI_Numba(self.mppi_cfg)
        self.mppi.setup(self.mppi_params)
        self.get_logger().info('MPPI NUMBA initialized.')

        # Publishers, subscribers, and tf
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        '''############### for costmap ##############'''
        self.local_costmap = None  # store the latest costmap
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, # Type: nav_msgs/msg/OccupancyGrid
            '/local_costmap/costmap',
            self.costmap_callback,
            1 # only the most recent message is kept in the queue
        )
        self.path_pub = self.create_publisher(Path, "/min_cost_path", 10)
        self.mppi_path_pub = self.create_publisher(Path, "/mppi_path", 10)

        self.no_costmap_received_timer = self.create_timer(2.0, self.notify_no_costmap)
        self.action_pub = self.create_publisher(
            msg_type=AckermannDriveStamped,
            topic="/drive",
            qos_profile=qos_profile_sensor_data,
        )

        # Create a timer to call the CUMPPI planner solve routine.
        self.timer = self.create_timer(0.1, self.solve_CUMPPI)
        self.i = 0
        self.isGoalReached = False
        # self.second_to_last_steering_angle = 0.0

    def notify_no_costmap(self):
        if self.local_costmap is None:
            self.get_logger().warn("No /local_costmap/costmap data received yet...")

    def costmap_callback(self, msg: OccupancyGrid):
        # Convert msg.data to a 2D list or np.array
        width = msg.info.width
        height = msg.info.height
        # Convert msg data to float costmap and store resolution/origin
        costmap_int8 = np.array(msg.data, dtype=np.int8).reshape(height, width)
        costmap_int8[costmap_int8 == -1] = 100 # make unknown area as obstacles 
        self.local_costmap = costmap_int8.astype(np.float32)
        self.cuniform_params['costmap_resolution'] = msg.info.resolution
        self.cuniform_params['costmap_origin'] = [msg.info.origin.position.x, msg.info.origin.position.y]
        self.cuniform_params['costmap'] = self.local_costmap

        # update MPPI parameters
        self.mppi_params['costmap_resolution'] = msg.info.resolution
        self.mppi_params['costmap_origin'] = [msg.info.origin.position.x, msg.info.origin.position.y]
        self.mppi_params['costmap'] = self.local_costmap
    
    def _dynamics_KS_3d_steering_angle(self, state, action, dt): #constant velocity
        x, y, theta = state
        steering_angle, v = action
        L_wb = 0.324 # wheelbase for F1Tenth
        x_new = x + v * np.cos(theta) * dt 
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + v/L_wb * np.tan(steering_angle) * dt
        return (x_new, y_new, theta_new) 
    
    def _state_to_pose(self, state):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(state[0])
        pose.pose.position.y = float(state[1])
        pose.pose.position.z = 0.0
        q = R.from_euler('z', float(state[2])).as_quat()
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        return pose

    def solve_CUMPPI(self):
        try:
            solve_CUMPPI_whole_start = time.perf_counter()
            # 1. Look up transform from map -> base_link
            transform = self.tf_buffer.lookup_transform(
                'map',           # source frame (or "map")
                # 'base_link',     # target frame (your robot)
                'laser',     # target frame (your robot)
                rclpy.time.Time()
            )
            # 2. Extract x, y
            x_robot = transform.transform.translation.x
            y_robot = transform.transform.translation.y
            # 3. Convert quaternion to yaw
            quat = transform.transform.rotation
            r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
            yaw_robot = r.as_euler('xyz', degrees=False)[2]            
            current_state = np.array([x_robot, y_robot, yaw_robot])

            # Update planner parameters with current state and latest costmap.
            self.cuniform_params['x0'] = current_state
            self.mppi_params['x0'] = current_state
            if self.local_costmap is not None:
                self.cuniform_params['costmap'] = self.local_costmap
                self.mppi_params['costmap'] = self.local_costmap

            # Update CUniform parameters with the latest data
            self.cuniform.shift_and_update(current_state, self.original_trajectories)
            self.cuniform.setup(self.cuniform_params)
            self.mppi.setup(self.mppi_params)

            refined_control = self.solve_CUMPPI_core(current_state)
            current_steering = refined_control[1]
            averaged_steering = current_steering

            # Log some status information
            if (self.i % 10) == 0:
                self.get_logger().info(f"  Whole solve_CUMPPI runtime {time.perf_counter()-solve_CUMPPI_whole_start}")
                self.get_logger().info(f'  Current configuration: x: {x_robot:.2f}, y: {y_robot:.2f}, theta: {yaw_robot:.2f}...')
                self.get_logger().info(f"  Target Position: x: {self.cuniform_params['xgoal'][0]}, y: {self.cuniform_params['xgoal'][1]}")
            if self.isGoalReached:
                refined_control = [0.0, 0.0]
                self.get_logger().info("Goal Reached!!!!")
            h = std_msgs.msg.Header()
            h.stamp = self.get_clock().now().to_msg()
            # Create and publish drive command based on refined_control.
            drive = AckermannDrive(steering_angle=float(averaged_steering), speed=float(refined_control[0]))
            data = AckermannDriveStamped(header=h, drive=drive)
            self.action_pub.publish(data)
            
            # Compute distance to goal and update goal status.
            dist2goal2 = (self.cuniform_params['xgoal'][0] - x_robot)**2 + (self.cuniform_params['xgoal'][1] - y_robot)**2
            goaltol2 = self.cuniform_params['goal_tolerance'] ** 2
            if dist2goal2 < goaltol2:
                self.isGoalReached = True
            else:
                self.isGoalReached = False
            self.i += 1
            if (self.i % 10) == 0:
                self.get_logger().info(f"  Distance to the Goal: {dist2goal2}, Goal Tolerance: {goaltol2}")
                self.get_logger().info('-------Iteration End----------')
        except Exception as e:
            tb_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
            self.get_logger().warn(f"Cannnot run solve_CUMPPI: {e}\n{tb_str}")

    def solve_CUMPPI_core(self, current_state):
        # Solve c_uniform planner.
        time_cuniform_start = time.perf_counter()
        min_idx, _, min_cost_trajectory, useq_numba = self.cuniform.solve()
        if (self.i % 10) == 0:
            self.get_logger().info(f"  runtime cuniform.solve(): {time.perf_counter() - time_cuniform_start}")

        u_seq_for_mppi = min_cost_trajectory[:, 3]
        constant_v = self.cuniform_params['vrange'][0]
        u_seq_for_mppi_vel = np.hstack((np.ones((u_seq_for_mppi.shape[0],1))*constant_v, u_seq_for_mppi.reshape(-1,1)))

        # set actions to the MPPI planner (dropping the final action)
        self.mppi.set_actions(u_seq_for_mppi_vel[:-1])
        # Solve MPPI to refine the control sequence.
        time_mppi = time.perf_counter()
        refined_control_sequence = self.mppi.solve()
        if (self.i % 10) == 0:
            self.get_logger().info(f"  runtime mppi.solve(): {time.perf_counter() - time_mppi}")
        self.mppi.shift_and_update(current_state, refined_control_sequence)

        time_publish = time.perf_counter()
        ############### visualize min cost traj below ##############
        visualize_traj = False
        if visualize_traj:
            pub_time = self.get_clock().now().to_msg()
            # Visualize minimum cost trajectory as a Path message
            path_msg = Path()
            path_msg.header.frame_id = "map"
            path_msg.header.stamp = pub_time
            for state in min_cost_trajectory:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = float(state[0])
                pose.pose.position.y = float(state[1])
                pose.pose.position.z = 0.0
                q = R.from_euler('z', float(state[2])).as_quat()
                pose.pose.orientation.x = q[0]
                pose.pose.orientation.y = q[1]
                pose.pose.orientation.z = q[2]
                pose.pose.orientation.w = q[3]
                path_msg.poses.append(pose)
            self.path_pub.publish(path_msg)

            # Visualize final MPPI trajectory by propagating the state using refined control sequence.
            mppi_path_msg = Path()
            mppi_path_msg.header.frame_id = "map"
            mppi_path_msg.header.stamp = pub_time
            propagated_state = current_state.copy()
            mppi_path_msg.poses.append(self._state_to_pose(propagated_state))
            for action in refined_control_sequence:
                # Swap action order: [v, steering_angle] -> (steering_angle, v)
                swapped_action = (action[1], action[0])
                propagated_state = self._dynamics_KS_3d_steering_angle(propagated_state, swapped_action, self.cfg.dt)
                mppi_path_msg.poses.append(self._state_to_pose(propagated_state))
            self.mppi_path_pub.publish(mppi_path_msg)
            if (self.i % 10) == 0:
                self.get_logger().info(f"  traj visualization runtime: {time.perf_counter() - time_publish}")
        return refined_control_sequence[0] # return the first control command 

    def on_shutdown(self):
        # Clean up the context for both numba and cuda
        self.get_logger().info('CUMPPI Planner Node shutting down')
        self.get_logger().info("Popping CUDA context...")
        primary_context.pop()
        primary_context.detach()

def main(args=None):
    rclpy.init(args=args)
    node = CUMPPIPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        rclpy.shutdown()
    return

if __name__ == "__main__":
    main()
    sys.exit(0)
