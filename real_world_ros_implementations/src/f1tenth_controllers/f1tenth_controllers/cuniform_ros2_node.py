#!/usr/bin/env python3
import rclpy
from tf2_ros import Buffer, TransformListener # for locolization
import traceback
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker
import numpy as np
from geometry_msgs.msg import TransformStamped  # Use TransformStamped instead of Rigids
from rclpy.qos import qos_profile_sensor_data
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import OccupancyGrid # for applying local costmap
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import time
import std_msgs
import pickle

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

DEFAULT_OBS_COST = 1e4
DEFAULT_DIST_WEIGHT = 10
np.set_printoptions(precision=2, suppress=True)

num_traj = 1000 # only use the first 'num_traj' trajectories
print("Reading cuniform trajectories...")
with open('/home/nvidia/f1tenth_ws/src/f1tenth_controllers/resource/FINAL_C_Uniform_100000_trajectories_disjoint_DUBINS_v_1_perturb_2.01_slack_2.01_seed_2025_grid_0.05_0.05_4.50deg_na45_t4.01_ts0.2.pkl', 'rb') as f:
    cuniform_trajectories = pickle.load(f)[:num_traj]

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

class Config:
  """ Configurations that are typically fixed throughout execution. """
  def __init__(self, 
               T=3, # Horizon (s)
               dt=0.2, # Length of each step (s)
               num_control_rollouts=16384, # Number of control sequences
               num_vis_state_rollouts=16384, # Number of visualization rollouts
               seed=1,
               ):
    
    self.seed = seed
    self.T = T
    self.dt = dt
    self.num_steps = int(T/dt)
    self.max_threads_per_block = max_threads_per_block # save just in case
    assert T > 0
    assert dt > 0
    assert T > dt
    assert self.num_steps > 0
    
    print('C-Uniform is used, could be either flow-based or neural c-uniform')

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


class CUniform_Numba(object):
  """ 
  Planner object that initializes GPU memory and runs CUniform on GPU via numba.   
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
    self.weights_d = None
    self.rng_states_d = None
    self.state_rollout_batch_d = None # For visualization only. Otherwise, inefficient

    # Other task specific params
    self.device_var_initialized = False
    self.generator = XORWOWRandomNumberGenerator()

    self.iteration_count = 0
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
    
    # Initialize all fixed-size device variables ahead of time. (Do not change in the lifetime of CUniform object)
    self.init_device_vars_before_solving()

  def load_trajectories(self, trajectories):
    self.trajectories = copy.deepcopy(trajectories)
    self.trajectories_d = numba_cuda.to_device(self.trajectories.astype(np.float32))

  def init_device_vars_before_solving(self):
    if not self.device_var_initialized:
      t0 = time.time()      
      self.costs_d = numba_cuda.device_array((self.num_control_rollouts), dtype=np.float32)
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
      print("CUniform parameters are not set. Cannot solve")
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
    goal_tolerance_d = np.float32(self.params['goal_tolerance'])

    vehicle_length_d = np.float32(self.vehicle_length)
    vehicle_width_d = np.float32(self.vehicle_width)
    vehicle_wheelbase_d = np.float32(self.vehicle_wheelbase)
    #obstacle
    obs_cost_d = np.float32(DEFAULT_OBS_COST if 'obs_penalty' not in self.params 
                                     else self.params['obs_penalty'])

    ''' COSTMAP Variables'''
    local_costmap_edt = self.params['costmap']
    local_costmap_edt = np.ascontiguousarray(local_costmap_edt)
    max_local_cost_d = np.float32(np.max(local_costmap_edt))
    # set the local costmap to the local_costmap_d on the device
    local_costmap_d = numba_cuda.to_device(local_costmap_edt)
    costmap_origin_x = np.float32(self.params['costmap_origin'][0])
    costmap_origin_y = np.float32(self.params['costmap_origin'][1])
    costmap_resolution = np.float32(self.params['costmap_resolution'])

    return xgoal_d, goal_tolerance_d, \
            vehicle_length_d, vehicle_width_d, vehicle_wheelbase_d, \
            obs_cost_d, local_costmap_d, max_local_cost_d, \
           costmap_origin_x, costmap_origin_y, costmap_resolution

  def get_rollout_cost(self):
    '''
    Calculate the cost of the each trajectories and find the trajectory with min cost. and return that trajectory to the host
    '''
    xgoal_d, goal_tolerance_d, \
    vehicle_length_d, vehicle_width_d, vehicle_wheelbase_d, \
    obs_cost_d, local_costmap_d, max_local_cost_d, \
    costmap_origin_x, costmap_origin_y, costmap_resolution \
    = self.move_cuniform_task_vars_to_device()
    
    # Weight for distance cost
    dist_weight = DEFAULT_DIST_WEIGHT if 'dist_weight' not in self.params else self.params['dist_weight']

    self.rollouts_cost_numba[self.num_control_rollouts, 1](
      self.trajectories_d,
      self.costs_d,
      goal_tolerance_d,
      xgoal_d,
      local_costmap_d, # local_costmap added
      max_local_cost_d,
      costmap_origin_x,
      costmap_origin_y,
      costmap_resolution,
      obs_cost_d,
      vehicle_length_d,
      vehicle_width_d,
      vehicle_wheelbase_d,
      dist_weight,
    )

    # Find the trajectory with the minimum cost
    min_cost_index = np.argmin(self.costs_d.copy_to_host())
    return min_cost_index, self.costs_d.copy_to_host(), self.trajectories_d[min_cost_index].copy_to_host()

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

  def control_dubins(self, x_curr, x_next, dt):
    '''Dubins Car model
    Forward simulate 
    x_curr[0] += dt_d*v_nom*math.cos(x_curr[2])
    x_curr[1] += dt_d*v_nom*math.sin(x_curr[2])
    x_curr[2] += dt_d*w_nom
    '''
    # Dubins Car model
    # calculate the control for w
    theta_curr = x_curr[2]
    theta_next = x_next[2]
    planner_dt = dt # 
    w = (theta_next - theta_curr)/planner_dt
    return w
  
  """GPU kernels from here on"""
  @staticmethod
  @numba_cuda.jit(fastmath=True)
  def rollouts_cost_numba(
    trajectories_d,
    costs_d,
    goal_tolerance_d,
    xgoal_d,
    local_costmap_d, # local_costmap added
    max_local_cost_d,
    costmap_origin_x,
    costmap_origin_y,
    params_costmap_resolution,
    obs_cost_d,
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
    goal_tolerance_d2 = goal_tolerance_d*goal_tolerance_d
    dist_to_goal2 = 1e9 # initialize to a large value

    x_curr = numba_cuda.local.array(3, numba.float32) # Dubins car model states x,y,theta
    x_curr_grid_d = numba_cuda.local.array((2), dtype=np.int32)

    gamma = 1.0 # Discount factor for cost
    # Loop through each state in the trajectory
    num_steps = trajectories_d.shape[1]
    for step in range(num_steps):
        # Extract current state (x, y, theta)
        for i in range(3):
          x_curr[i] = trajectories_d[bid, step, i]        

        # Compute distance to goal
        dist_to_goal2 = (xgoal_d[0]-x_curr[0])**2 + (xgoal_d[1]-x_curr[1])**2
        costs_d[bid] += stage_cost(dist_to_goal2, dist_weight_d) * gamma

        convert_position_to_costmap_indices_gpu(
            x_curr[0],
            x_curr[1],
            costmap_origin_x,
            costmap_origin_y,
            params_costmap_resolution,
            x_curr_grid_d,
        )
        costs_d[bid] += calculate_localcostmap_cost(local_costmap_d, x_curr_grid_d) * obs_cost_d * gamma

        if dist_to_goal2  <= goal_tolerance_d2:
          goal_reached = True
          break
      
    # Accumulate terminal cost 
    costs_d[bid] += term_cost(dist_to_goal2, goal_reached)

class CUniformPlannerNode(Node):
    def __init__(self):
        super().__init__('cuniform_planner_node')

        self.cfg = Config(T = 3,
            dt = 0.2,
            num_control_rollouts = 1000, # Same as number of blocks, can be more than 1024
            num_vis_state_rollouts = 1000,
            seed = 1,
        )
        
        self.cuniform = CUniform_Numba(self.cfg)
        self.original_trajectories = cuniform_trajectories_transformed
        self.u_execute = [0.0, 0.0] # initial action

        # CUniform initial parameters
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

          # Control and sample specification
          vrange = np.array([1.0, 1.0]), # Linear velocity range. Constant Linear Velocity
          
          costmap = None, # intiallly nothing
          obs_penalty = 1e4
        )
        self.path_pub = self.create_publisher(Path, "/min_cost_path", 10)

        self.lookahead_marker_pub = self.create_publisher(Marker, "/lookahead_marker", 5)
        self.lookahead_marker_timer = self.create_timer(0.1, self.lookahead_publish_waypoint)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        '''############### for costmap ##############'''
        # high level steps
        # Step 1: subscribe to OccupancyGrid and convert msg.data into a numpy array
        # Step 2: Pass that array to cuniform_params['costmap'] in solve_cuniform()
        self.local_costmap = None  # store the latest costmap
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, # Type: nav_msgs/msg/OccupancyGrid
            '/local_costmap/costmap',
            self.costmap_callback,
            1 # only the most recent message is kept in the queue
        )
        self.debug_local_costmap_pub = self.create_publisher(OccupancyGrid, '/debug_local_costmap', 1)
        self.no_costmap_received_timer = self.create_timer(2.0, self.notify_no_costmap)

        self.action_pub = self.create_publisher(
            msg_type=AckermannDriveStamped,
            topic="/drive",
            qos_profile=qos_profile_sensor_data,
        )

        # Create a timer to call the CUniform solver every 100ms (0.1s)
        self.timer = self.create_timer(0.125, self.solve_cuniform)
        self.i = 0
        self.isGoalReached = False

        self.cuniform.setup(self.cuniform_params)
        
        self.cuniform.load_trajectories(self.original_trajectories[:])
        self.get_logger().info('Cuniform Planner Node started')

    def lookahead_publish_waypoint(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lookahead_waypoint"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.cuniform_params['xgoal'][0]
        marker.pose.position.y = self.cuniform_params['xgoal'][1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25

        marker.color.a = 1.0
        marker.color.r = 1.0
        self.lookahead_marker_pub.publish(marker)

    def notify_no_costmap(self):
        if self.local_costmap is None:
            self.get_logger().warn("No /local_costmap/costmap data received yet...")

    def publish_local_costmap_debug(self):
        if self.cuniform.params['costmap'] is not None:
            height, width = self.cuniform.params['costmap'].shape

            msg = OccupancyGrid()
            msg.header.frame_id = "map"
            msg.header.stamp = self.get_clock().now().to_msg()

            msg.info.width = width
            msg.info.height = height
            msg.info.resolution = self.cuniform.params['costmap_resolution']

            # Shift the origin so the costmap is centered on local_costmap_origin
            msg.info.origin.position.x = self.cuniform.local_costmap_origin[0]
            msg.info.origin.position.y = self.cuniform.local_costmap_origin[1] 

            # Convert float costmap to int8
            costmap_int8 = self.cuniform.params['costmap'].astype(np.int8).flatten()
            msg.data = costmap_int8.tolist()
            self.debug_local_costmap_pub.publish(msg)

    def costmap_callback(self, msg: OccupancyGrid):
        # Convert msg.data to a 2D list or np.array
        width = msg.info.width
        height = msg.info.height
        # Convert msg data to float costmap and store resolution/origin
        costmap_int8 = np.array(msg.data, dtype=np.int8).reshape(height, width)
        # costmap_int8[costmap_int8 == -1] = 100 # make unknown area as obstacles 
        self.local_costmap = costmap_int8.astype(np.float32)
        self.cuniform_params['costmap_resolution'] = msg.info.resolution
        self.cuniform_params['costmap_origin'] = [msg.info.origin.position.x, msg.info.origin.position.y]
        self.cuniform_params['costmap'] = self.local_costmap
        
    def solve_cuniform(self):
        try:
            c_uniform_start = time.time()
            # 1. Look up transform from map -> base_link
            transform = self.tf_buffer.lookup_transform(
                'map',           # source frame (or "map")
                'base_link',     # target frame (your robot)
                rclpy.time.Time()
            )
            # 2. Extract x, y
            x_robot = transform.transform.translation.x
            y_robot = transform.transform.translation.y
            # 3. Convert quaternion to yaw
            quat = transform.transform.rotation
            r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
            yaw_robot = r.as_euler('xyz', degrees=False)[2]            

            self.cuniform_params['x0'] = np.array([x_robot, y_robot, yaw_robot])
            if self.local_costmap is not None:
                self.cuniform_params['costmap'] = self.local_costmap

            # Update CUniform parameters with the latest data
            x_current = np.array([x_robot, y_robot, yaw_robot])
            self.cuniform.shift_and_update(x_current, self.original_trajectories)
            self.cuniform.setup(self.cuniform_params)

            ''' Solve CUniform to get the minimum cost trajectory '''
            min_idx, _, min_traj = self.cuniform.solve()
              
            ''' Get the first time step control through inverse dynamic'''
            omega = self.cuniform.control_dubins(x_current, self.cuniform.trajectories[min_idx][1], dt=0.2)
            # NOTE: if using kinematic model or neural cuniform, you can directly uses the omega
            # omega = self.cuniform.trajectories[min_idx][0][3]
            self.u_execute = [1.0, omega]

            ############### visualize min cost traj below ##############
            # Visualize minimum cost trajectory as a Path message
            path_msg = Path()
            path_msg.header.frame_id = "map"
            path_msg.header.stamp = self.get_clock().now().to_msg()
            for state in min_traj:
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
            ############### visualize min cost traj above ##############
            self.cuniform.local_costmap_origin = self.cuniform_params['costmap_origin']
            self.publish_local_costmap_debug()
           
            if self.i > 3:
              h = std_msgs.msg.Header()
              h.stamp = self.get_clock().now().to_msg()
              if self.isGoalReached: 
                self.u_execute = [0.0, 0.0]
                drive = AckermannDrive(steering_angle=self.u_execute[0], speed=self.u_execute[1])
                data = AckermannDriveStamped(header=h, drive=drive)
                self.get_logger().info(f"Goal Reached!!!!")
              else:   
                ''' # NOTE: if using Dubin's model, you need to do some conversion to make it aligns with the F1Tenth's dynamic
                drive = AckermannDrive(
                  steering_angle=0.9*np.arctan2(
                     (self.cuniform_params['vehicle_wheelbase'])*self.u_execute[1], self.u_execute[0]
                    ), speed=1.0
                  )
                '''
                # NOTE: if using kinematic model, you can directly uses the omega
                drive = AckermannDrive(steering_angle=self.u_execute[1], speed=1.0)
                data = AckermannDriveStamped(header=h, drive=drive)

              if ((self.i % 10) == 0): 
                self.get_logger().info(f"Input given: velocity {self.u_execute[0]}, Steering_Angle: {np.rad2deg(self.u_execute[1]*1.0)}" )
              self.action_pub.publish(data)
              if ((self.i % 10) == 0): 
                self.get_logger().info(f'-----------------')
                self.get_logger().info(f'Running CUniform solver with configuration: x: {x_robot:.2f}, y: {y_robot:.2f}, theta: {yaw_robot:.2f}...')
                self.get_logger().info(f"Target Position: x: {self.cuniform_params['xgoal'][0]}, z: {self.cuniform_params['xgoal'][1]}")
                self.get_logger().info(f'Runtime for solve_cuniform: {time.time() - c_uniform_start}')

              dist2goal2 = (self.cuniform_params['xgoal'][0] - x_robot)**2 + (self.cuniform_params['xgoal'][1] - y_robot)**2
              
              goaltol2 = self.cuniform_params['goal_tolerance'] * self.cuniform_params['goal_tolerance']
              if ((self.i % 10) == 0): 
                self.get_logger().info(f"Distance to the Goal: {dist2goal2}, Goal Tolerance: {goaltol2}")
              if dist2goal2 < goaltol2:
                self.isGoalReached = True
              if dist2goal2 > goaltol2:
                self.isGoalReached = False
            self.i += 1
        except Exception as e:
            tb_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
            self.get_logger().warn(f"Cannnot run solve_cuniform: {e}\n{tb_str}")

    def on_shutdown(self):
        self.get_logger().info('CUniform Planner Node shutting down')
        self.get_logger().info("Popping CUDA context...")
        # Clean up the context for both numba and cuda
        primary_context.pop()
        primary_context.detach()

def main(args=None):
    rclpy.init(args=args)
    node = CUniformPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        rclpy.shutdown()
if __name__ == '__main__':
    main()
