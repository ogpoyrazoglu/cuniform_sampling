#!/usr/bin/env python3
from tf2_ros import Buffer, TransformListener # for locolization
import rclpy
import traceback
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_cdt, distance_transform_edt
from geometry_msgs.msg import TransformStamped  # Use TransformStamped instead of Rigids
from collections import deque  # For implementing a circular buffer
from rclpy.qos import qos_profile_sensor_data
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid # for applying local costmap
import std_msgs
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.curandom import XORWOWRandomNumberGenerator
from pycuda import gpuarray

cuda.init()
device = cuda.Device(0)
primary_context = device.retain_primary_context()
primary_context.push()

from numba import cuda as numba_cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
from .cuda_device_functions import *
from .input_constraints import *

gpu = numba_cuda.get_current_device()
print(numba_cuda.is_available())
max_threads_per_block = gpu.MAX_THREADS_PER_BLOCK
max_square_block_dim = (int(gpu.MAX_BLOCK_DIM_X**0.5), int(gpu.MAX_BLOCK_DIM_X**0.5))
max_blocks = gpu.MAX_GRID_DIM_X
max_rec_blocks = rec_max_control_rollouts = int(1e6) # Though theoretically limited by max_blocks on GPU
rec_min_control_rollouts = 100

np.set_printoptions(precision=2, suppress=True)

DEFAULT_OBS_COST = 1e4
ACTION_WEIGHT = 10.0

class Config:
  """ Configurations that are typically fixed throughout execution. """
  def __init__(self, 
               T=5, # Horizon (s)
               dt=0.1, # Length of each step (s)
               num_control_rollouts=16384, # Number of control sequences
               num_vis_state_rollouts=16384, # Number of visualization rollouts
               seed=1,
               mppi_type=0): # Normal dist / 1: NLN):
    self.seed = seed
    self.T = T
    self.dt = dt
    self.num_steps = int(T/dt)
    self.max_threads_per_block = max_threads_per_block # save just in case
    self.mppi_type = mppi_type # Normal dist / 1: NLN
    assert T > 0
    assert dt > 0
    assert T > dt
    assert self.num_steps > 0
    
    if self.mppi_type == 0:
      print('Vanilla MPPI is used')
    else:
      print('log-MPPI is used')

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
    self.mppi_type = self.cfg.mppi_type # Normal dist / 1: NLN
    if self.mppi_type == 1:
      # print("NLN is used for noise")
      self.mu_LogN, self.std_LogN = Normal2LogN(0, np.mean([0.1, 0.1]))
      print('the mu:', self.mu_LogN)
      print('the std:', self.std_LogN)
      self.LogN_info = [self.mppi_type, self.mu_LogN, self.std_LogN]

    # local costmap size and resolution/
    self.local_costmap_size = 120
    self.costmap_loaded = False
    self.reset()

  def reset(self):
    # Other task specific params
    self.u_seq0 = np.zeros((self.num_steps, 2), dtype=np.float32)
    self.u_seq0[:,0] = 1.0 # Linear velocity
    self.params = None
    self.params_set = False
    self.u_prev_d = None
    self.costmap_loaded = False
    
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
      
      self.debug_d = numba_cuda.device_array((self.num_control_rollouts,self.num_steps+1, 4), dtype=np.float32)

      self.state_rollout_batch_d = numba_cuda.device_array((self.num_vis_state_rollouts, self.num_steps+1, 3), dtype=np.float32) # 3: x, y, theta  
      self.local_costmap_d = numba_cuda.device_array((self.local_costmap_size, self.local_costmap_size), dtype=np.float32)   
      self.device_var_initialized = True
      print("MPPI planner has initialized GPU memory after {} s".format(time.time()-t0))

  def setup(self, params):
    # These tend to change (e.g., current robot position, the map) after each step
    self.set_params(params)

  def set_params(self, params):
    self.params = copy.deepcopy(params)
    self.params_set = True
    if self.params['costmap'] is not None: # should be type ndarray
        self.costmap_loaded = True

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
    if self.mppi_type == 1: # NLN
        # print('NLN IS USED FOR NOISE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
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
           u_std_d, x0_d, dt_d, local_costmap_d, obs_cost_d, max_local_cost_d, \
           costmap_origin_x, costmap_origin_y, costmap_resolution

  def solve_with_nominal_dynamics(self):
    """
    Launch GPU kernels that use nominal dynamics but adjsuts cost function based on worst-case linear speed.
    """
    vrange_d, wrange_d, xgoal_d, \
      goal_tolerance_d, lambda_weight_d, \
      vehicle_length_d, vehicle_width_d, vehicle_wheelbase_d,\
      u_std_d, x0_d, dt_d, \
      local_costmap_d, obs_cost_d, max_local_cost_d,\
      costmap_origin_x, costmap_origin_y, params_costmap_resolution = self.move_mppi_task_vars_to_device()
  
    # Weight for distance cost
    dist_weight = 1e4 if 'dist_weight' not in self.params else self.params['dist_weight']

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
      self.rollout_numba[self.num_control_rollouts, 1](
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
        self.noise_samples_d,
        self.u_cur_d,
        # results
        self.costs_d,
        costmap_origin_x,
        costmap_origin_y,
        params_costmap_resolution,
        self.debug_d        
      )
      self.u_prev_d = self.u_cur_d

      debug_arr = self.debug_d.copy_to_host()
      # print(f'Debug ARR {np.array(debug_arr[0,:,:])}')
      # print(f'Debug ARR {np.array(debug_arr[1,:,:])}')
      # print(f'Debug ARR {np.array(debug_arr[2,:,:])}')

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
    # self.params["x0"] = new_x0.copy()
    self.shift_optimal_control_sequence(u_cur, num_shifts)

  def shift_optimal_control_sequence(self, u_cur, num_shifts=1):
    u_cur_shifted = u_cur.copy()
    u_cur_shifted[:-num_shifts] = u_cur_shifted[num_shifts:]
    self.u_cur_d = numba_cuda.to_device(u_cur_shifted.astype(np.float32))

  def get_state_rollout(self):
    """
    Generate state sequences based on the current optimal control sequence.
    """
    assert self.params_set, "MPPI parameters are not set"
    if not self.device_var_initialized:
      print("Device variables not initialized. Cannot run mppi.")
      return
    # Move things to GPU
    vrange_d = numba_cuda.to_device(self.params['vrange'].astype(np.float32))
    wrange_d = numba_cuda.to_device(self.params['wrange'].astype(np.float32))
    x0_d = numba_cuda.to_device(self.params['x0'].astype(np.float32))
    dt_d = np.float32(self.params['dt'])
    vehicle_wheelbase_d = np.float32(self.vehicle_wheelbase)

    self.get_state_rollout_across_control_noise[self.num_vis_state_rollouts, 1](
        self.state_rollout_batch_d, # where to store results
        x0_d, 
        dt_d,
        self.noise_samples_d,
        vrange_d,
        wrange_d,
        vehicle_wheelbase_d,
        self.u_prev_d,
        self.u_cur_d,
        )
    return self.state_rollout_batch_d.copy_to_host()

  def get_vehicle_boundary_points_p(self, x_curr, vehicle_length, vehicle_width):
    x_center, y_center, theta = x_curr
    # Half dimensions
    half_length = vehicle_length / 2
    half_width = vehicle_width / 2

    # Define the relative positions of the corners
    corners = np.array([
        [half_length, half_width],     # Front left
        [half_length, -half_width],    # Front right
        [0, -half_width],               # center right,   
        [-half_length, -half_width],   # Rear right
        [-half_length, half_width],    # Rear left,
        [0, half_width]    # center left
    ])

    # Compute the rotation matrix based on heading angle (theta)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])

    # Rotate corners by the heading angle and translate to world coordinates
    world_corners = rotation_matrix @ corners.T 
    world_corners = world_corners.T + np.array([x_center, y_center])
    # Add first point to the end for visualization
    world_corners = np.vstack([world_corners, world_corners[0]])
    return world_corners

  @staticmethod
  @numba_cuda.jit(fastmath=True)
  def rollout_numba(
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
          dist_weight_d,
          noise_samples_d,
          u_cur_d,
          costs_d,
          costmap_origin_x,
          costmap_origin_y,
          params_costmap_resolution,
          debug_d
      ):
    """
    There should only be one thread running in each block, where each block handles a single sampled control sequence.
    """

    # Get block id and thread id
    bid = numba_cuda.blockIdx.x   # index of block
    tid = numba_cuda.threadIdx.x  # index of thread within a block
    costs_d[bid] = 0.0

    # Explicit unicycle update and map lookup
    # From here on we assume grid is properly padded so map lookup remains valid
    x_curr = numba_cuda.local.array(3, numba.float32) # Dubins car model states x,y,theta
    local_costmap_origin = numba_cuda.local.array(2, numba.float32) # x, y
    x_curr_grid_d = numba_cuda.local.array((2), dtype=np.int32)
    for i in range(3): 
      x_curr[i] = x0_d[i]
    for i in range(2):
      local_costmap_origin[i] = x0_d[i]
    timesteps = len(u_cur_d)

    goal_reached = False
    isCollided = False
    goal_tolerance_d2 = goal_tolerance_d*goal_tolerance_d
    dist_to_goal2 = 1e9 # initialize to a large value

    v_nom = v_noisy = w_nom = w_noisy = 0.0
    gamma = 1.0 # Discount factor for cost

    for t in range(timesteps):
      # Nominal noisy control
      v_noisy = v_nom = 1.0 # Constant velocity
      w_nom = u_cur_d[t, 1] + noise_samples_d[bid, t, 1]
      w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))
      # Forward simulate
      # kinematic model update
      x_curr[0] += dt_d*v_noisy*math.cos(x_curr[2])
      x_curr[1] += dt_d*v_noisy*math.sin(x_curr[2])
      x_curr[2] += dt_d*v_noisy*math.tan(w_noisy)/vehicle_wheelbase_d

      # Get current state costmap indices
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
        # Compute distance to goal
        dist_to_goal2 = (((xgoal_d[0]-x_curr[0])**2 + (xgoal_d[1]-x_curr[1])**2))**0.5
        costs_d[bid]+= stage_cost(dist_to_goal2, 5.0)

        if dist_to_goal2 <= goal_tolerance_d:
          goal_reached = True
          break
        prev_dist_to_goal2 = dist_to_goal2
      else:
        costs_d[bid] += stage_cost(prev_dist_to_goal2, 5.0) # distance to goal cost
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

  @staticmethod
  @numba_cuda.jit(fastmath=True)
  def get_state_rollout_across_control_noise(
          state_rollout_batch_d, # where to store results
          x0_d, 
          dt_d,
          noise_samples_d,
          vrange_d,
          wrange_d,
          vehicle_wheelbase_d,
          u_prev_d,
          u_cur_d):
    """
    Do a fixed number of rollouts for visualization across blocks.
    Assume kernel is launched as get_state_rollout_across_control_noise[num_blocks, 1]
    The block with id 0 will always visualize the best control sequence. Other blocks will visualize random samples.
    """
    
    # Use block id
    tid = numba_cuda.threadIdx.x
    bid = numba_cuda.blockIdx.x
    timesteps = len(u_cur_d)

    if bid==0:
      # Visualize the current best 
      # Explicit unicycle update and map lookup
      # From here on we assume grid is properly padded so map lookup remains valid
      x_curr = numba_cuda.local.array(3, numba.float32) # x, y, theta

      for i in range(3): 
        x_curr[i] = x0_d[i]
        state_rollout_batch_d[bid,0,i] = x0_d[i]
      
      for t in range(timesteps):
        # Nominal noisy control
        v_nom = 1.0
        w_nom = u_cur_d[t, 1]
        # Forward simulate
        # dubins car model update
        x_curr[2] += dt_d*w_nom
        x_curr[2] = math.fmod(x_curr[2], 2*math.pi)
        x_curr[0] += dt_d*v_nom*math.cos(x_curr[2])
        x_curr[1] += dt_d*v_nom*math.sin(x_curr[2])

        # Save state x, y, theta
        state_rollout_batch_d[bid,t+1,0] = x_curr[0]
        state_rollout_batch_d[bid,t+1,1] = x_curr[1]
        state_rollout_batch_d[bid,t+1,2] = x_curr[2]
    
    else:
      # Explicit unicycle update and map lookup
      # From here on we assume grid is properly padded so map lookup remains valid
      x_curr = numba_cuda.local.array(3, numba.float32)
      for i in range(3): 
        x_curr[i] = x0_d[i]
        state_rollout_batch_d[bid,0,i] = x0_d[i]

      for t in range(timesteps):
        # Nominal noisy control
        # v_nom = u_prev_d[t, 0] + noise_samples_d[bid, t, 0]
        w_nom = u_prev_d[t, 1] + noise_samples_d[bid, t, 1]
        v_noisy = 1.0
        w_noisy = max(wrange_d[0], min(wrange_d[1], w_nom))

        # Forward simulate
        x_curr[2] += dt_d*w_noisy
        x_curr[2] = math.fmod(x_curr[2], 2*math.pi)
        x_curr[0] += dt_d*v_noisy*math.cos(x_curr[2])
        x_curr[1] += dt_d*v_noisy*math.sin(x_curr[2])

        # Save state x, y, theta
        state_rollout_batch_d[bid,t+1,0] = x_curr[0]
        state_rollout_batch_d[bid,t+1,1] = x_curr[1]
        state_rollout_batch_d[bid,t+1,2] = x_curr[2]

  @staticmethod
  @numba_cuda.jit(fastmath=True)
  def sample_noise_numba(rng_states, u_std_d, noise_samples_d):
    """
    Should be invoked as sample_noise_numba[NUM_U_SAMPLES, NUM_THREADS].
    noise_samples_d.shape is assumed to be (num_rollouts, time_steps, 2)
    Assume each thread corresponds to one time step
    For consistency, each block samples a sequence, and threads (not too many) work together over num_steps.
    This will not work if time steps are more than max_threads_per_block (usually 1024)
    """
    block_id = numba_cuda.blockIdx.x
    thread_id = numba_cuda.threadIdx.x
    abs_thread_id = numba_cuda.grid(1)

    noise_samples_d[block_id, thread_id, 0] = u_std_d[0]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)
    noise_samples_d[block_id, thread_id, 1] = u_std_d[1]*xoroshiro128p_normal_float32(rng_states, abs_thread_id)

class MPPIPlannerNode(Node):
    def __init__(self):
        super().__init__('mppi_planner_node')

        # Initialize configuration for MPPI
        self.cfg = Config(T = 3,
            dt = 0.2,
            num_control_rollouts =1500, # Same1 as number of blocks, can be more than 1024
            num_vis_state_rollouts = 500,
            seed = 1,
            mppi_type = 1
          )
        self.mppi = MPPI_Numba(self.cfg)
        self.map_path = "/home/nvidia/f1tenth_ws/src/pure_pursuit/racelines/shepherd_lab_raceline_v1.csv"
        data = np.loadtxt(self.map_path, delimiter = ",")

        # MPPI initial parameters
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
          # variance = 0.1
          u_std = np.array([0.023, 0.10]), # Noise std for sampling linear and angular velocities.
          vrange = np.array([1.0, 1.0]), # Linear velocity range. Constant Linear Velocity
          wrange = np.array([-np.pi/6, np.pi/6]), # Angular velocity range.
          costmap = None, # intiallly nothing
          obs_penalty = 1e2
        )

        '''############### for path following ###############'''
        self.cx = data[:, 0] # 1st column of data -> x-position of the waypoints
        self.cy = data[:, 1] # 2nd column of data -> y-position of the waypoints
        self.cv = data[:, 2] # 3rd column of data -> velocity of the waypoints
        
        self.rear_x = self.mppi_params['x0'][0] - ((self.mppi_params['vehicle_wheelbase'] / 2) * math.cos(self.mppi_params['x0'][2]))
        self.rear_y = self.mppi_params['x0'][1] - ((self.mppi_params['vehicle_wheelbase'] / 2) * math.sin(self.mppi_params['x0'][2]))

        self.min_lookahead = 1.0
        self.max_lookahead = 3.0
        self.lookahead_ratio = 1.5
        self.current_index = None
        self.target_index = 0

        self.mppi_path_pub = self.create_publisher(Path, "/mppi_path", 10)

        # self publish the marker array
        self.lookahead_marker_pub = self.create_publisher(Marker, "/lookahead_marker", 5)
        self.lookahead_marker_timer = self.create_timer(0.1, self.lookahead_publish_waypoint)
        self.curr_marker_pub = self.create_publisher(Marker, "/curr_marker", 5)
        self.currmarker_timer = self.create_timer(0.1, self.curr_publish_waypoint)
        
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
        self.debug_local_costmap_pub = self.create_publisher(OccupancyGrid, '/debug_local_costmap', 1)
        self.action_pub = self.create_publisher(
            msg_type=AckermannDriveStamped,
            topic="/drive",
            qos_profile=qos_profile_sensor_data,
        )
        # Create a timer to call the MPPI solver every 100ms (0.1s)
        self.timer = self.create_timer(0.1, self.solve_mppi)
        self.i = 0
        self.isGoalReached = False
        self.mppi.setup(self.mppi_params)
        self.get_logger().info('MPPI Planner Node started')

    def publish_local_costmap_debug(self):
        if self.mppi.params['costmap'] is not None:
            height, width = self.mppi.params['costmap'].shape
            
            msg = OccupancyGrid()
            msg.header.frame_id = "map"
            msg.header.stamp = self.get_clock().now().to_msg()
            
            msg.info.width = width
            msg.info.height = height
            msg.info.resolution = self.mppi.params['costmap_resolution']
            
            # Shift the origin so the costmap is centered on local_costmap_origin
            msg.info.origin.position.x = self.mppi.local_costmap_origin[0]
            msg.info.origin.position.y = self.mppi.local_costmap_origin[1] 
            
            # Convert float costmap to int8
            costmap_int8 = self.mppi.params['costmap'].astype(np.int8).flatten()
            msg.data = costmap_int8.tolist()
            self.debug_local_costmap_pub.publish(msg)

    def costmap_callback(self, msg: OccupancyGrid):
        # Convert msg.data to a 2D list or np.array
        width = msg.info.width
        height = msg.info.height
        # Convert msg data to float costmap and store resolution/origin
        costmap_int8 = np.array(msg.data, dtype=np.int8).reshape(height, width)
        costmap_int8[costmap_int8 == -1] = 100 # make unknown area as obstacles 
        self.local_costmap = costmap_int8.astype(np.float32)
        self.mppi_params['costmap_resolution'] = msg.info.resolution
        self.mppi_params['costmap_origin'] = [msg.info.origin.position.x, msg.info.origin.position.y]

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)
    
    def lookahead_publish_waypoint(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lookahead_waypoint"
        # self.get_logger().info(f'Target Waypoint id: {self.target_index}')
        # marker.id = int(str(self.target_index))
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        # marker.pose.position.x = self.cx[self.target_index]
        # marker.pose.position.y = self.cy[self.target_index]
        marker.pose.position.x = -1.0
        marker.pose.position.y = -15.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25

        marker.color.a = 1.0
        marker.color.r = 1.0

        self.lookahead_marker_pub.publish(marker)

    def curr_publish_waypoint(self):
        index = 0
        if self.current_index == None:
            index = 1
        else:
             index = self.current_index
        # self.get_logger().info(f'Current Waypoint id: {self.current_index}')
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "curr_waypoint"
        # marker.id = int(str(self.current_index))
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = self.cx[index]
        marker.pose.position.y = self.cy[index]

        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25

        marker.color.a = 1.0
        marker.color.b = 0.0

        self.curr_marker_pub.publish(marker)

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

    def search_target_index(self):
        # To speed up nearest point search, doing it at only first time.
        if self.current_index is None:
            # search nearest point index
            dx = [self.rear_x - icx for icx in self.cx]
            dy = [self.rear_y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.current_index = ind
        else:
            ind = self.current_index
            while True:
                distance_this_index = self.calc_distance(self.cx[ind], self.cy[ind])
                distance_next_index = self.calc_distance(self.cx[(ind + 1) % len(self.cx)], self.cy[(ind + 1) % len(self.cy)])
                if distance_this_index < distance_next_index:
                    break
                ind = (ind + 1) % len(self.cx)  # Ensure wrap-around in a circular path
            self.current_index = ind

        current_velocity = self.mppi_params['vrange'][0] # assume constant velocity
        Lf = min(max(self.min_lookahead, self.max_lookahead * current_velocity / self.lookahead_ratio), self.max_lookahead)
        if self.i % 20 == 0:
            self.get_logger().info(f'Lookahead Distance: {Lf}, Current Velocity: {current_velocity}')
            dist_to_next = self.calc_distance(self.cx[ind], self.cy[ind])
            self.get_logger().info(f'Distance to next waypoint: {dist_to_next}')
        # search look ahead target point index
        while Lf > self.calc_distance(self.cx[ind], self.cy[ind]):
            ind = (ind + 1) % len(self.cx)  # Wrap around for circular paths
            if ind == self.current_index:  # Avoid infinite loop in case of very small Lf
                break

        return ind, Lf

    def solve_mppi(self):
        try:
            # 1. Look up transform from map -> base_link
            transform = self.tf_buffer.lookup_transform(
                'map',           # source frame 
                'base_link',     # target frame 
                #'laser',     
                rclpy.time.Time()
            )

            # 2. Extract x, y
            x_robot = transform.transform.translation.x
            y_robot = transform.transform.translation.y

            # 3. Convert quaternion to yaw
            quat = transform.transform.rotation
            r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
            yaw_robot = r.as_euler('xyz', degrees=False)[2]            

            # 4. Update the MPPI initial state
            self.mppi_params['x0'] = np.array([x_robot, y_robot, yaw_robot])
            self.rear_x = self.mppi_params['x0'][0] - ((self.mppi_params['vehicle_wheelbase'] / 2) * math.cos(self.mppi_params['x0'][2]))
            self.rear_y = self.mppi_params['x0'][1] - ((self.mppi_params['vehicle_wheelbase'] / 2) * math.sin(self.mppi_params['x0'][2]))

            # If we have a valid costmap, pass it to MPPI
            if self.local_costmap is not None:
                self.mppi_params['costmap'] = self.local_costmap
            
            ind = self.search_target_index()[0]
            if self.target_index >= ind:
                ind = self.target_index
            self.target_index = ind
            # global_tx = self.cx[ind] # This is the target waypoints x position
            # global_ty = self.cy[ind] # This is the target waypoints y position
            # latest_target_pos = [global_tx, global_ty]
            # self.mppi_params['xgoal'] = np.array([latest_target_pos[0], latest_target_pos[1]])
            self.mppi.setup(self.mppi_params)
            self.mppi.local_costmap_origin = self.mppi_params['costmap_origin']

            # Solve MPPI
            result = self.mppi.solve()
            mppi_path_msg = Path()
            mppi_path_msg.header.frame_id = "map"
            mppi_path_msg.header.stamp = self.get_clock().now().to_msg()
            propagated_state = self.mppi_params['x0'].copy()
            mppi_path_msg.poses.append(self._state_to_pose(propagated_state))
            for action in result:
                # Swap action order: [v, steering_angle] -> (steering_angle, v)
                swapped_action = (action[1], action[0])
                propagated_state = self._dynamics_KS_3d_steering_angle(propagated_state, swapped_action, self.cfg.dt)
                mppi_path_msg.poses.append(self._state_to_pose(propagated_state))
            self.mppi_path_pub.publish(mppi_path_msg)

            self.publish_local_costmap_debug()

            #get the first action 
            u_execute = result[0]
            if self.i > 3:
              h = std_msgs.msg.Header()
              h.stamp = self.get_clock().now().to_msg()
              if ((self.i % 10) == 0): 
                self.get_logger().info(f"Input given: velocity {u_execute[0]}, Steering_Angle: {np.rad2deg(u_execute[1])}" )
                self.get_logger().info(f"Target Position: x: {self.mppi_params['xgoal'][0]}, z: {self.mppi_params['xgoal'][1]}")
                self.get_logger().info(f"F1tenth Configuration x: {x_robot}, y:{y_robot}, yaw: {yaw_robot}")

              if self.isGoalReached:
                u_execute = [0.0, 0.0]
                drive = AckermannDrive(steering_angle=float(u_execute[1]), speed=float(u_execute[0]))
                data = AckermannDriveStamped(header=h, drive=drive)
                self.get_logger().info(f"Goal Reached!!!!")
              else: 
                drive = AckermannDrive(steering_angle=float(u_execute[1]), speed=1.0)
                data = AckermannDriveStamped(header=h, drive=drive)
              self.action_pub.publish(data)
              self.mppi.shift_and_update(self.mppi_params['x0'], result, 1)

              dist2goal2 = (self.mppi_params['xgoal'][0] - x_robot)**2 \
                         + (self.mppi_params['xgoal'][1] - y_robot)**2
              
              goaltol2 = self.mppi_params['goal_tolerance'] * self.mppi_params['goal_tolerance']
              if ((self.i % 10) == 0): 
                self.get_logger().info(f"Distance to the Goal: {np.sqrt(dist2goal2)}, Goal Tolerance: {goaltol2}, Current target_index: {self.target_index}")
              if dist2goal2 < goaltol2:
                self.isGoalReached = True
                self.target_index  = self.search_target_index()[0]
              if dist2goal2 > goaltol2:
                self.isGoalReached = False
            self.i += 1

        except Exception as e:
            tb_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
            self.get_logger().warn(f"Could not lookup TF transform: {e}\n{tb_str}")
            return

    def on_shutdown(self):
        self.get_logger().info('MPPI Planner Node shutting down')
        self.get_logger().info("Popping CUDA context...")
        # Clean up the context for both numba and cuda
        primary_context.pop()
        primary_context.detach()

def main(args=None):
    rclpy.init(args=args)
    node = MPPIPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        rclpy.shutdown()
if __name__ == '__main__':
    main()
