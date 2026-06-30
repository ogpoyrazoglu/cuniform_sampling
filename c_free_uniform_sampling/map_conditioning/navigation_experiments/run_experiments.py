"""Main script to run BARN navigation experiments."""
import os
import yaml
import argparse
from datetime import datetime
import gc
import torch
import re

from core.environments_manager import EnvironmentsManager
from controllers.mppi_pytorch_controller import MPPIPyTorchController
from controllers.nn_cuniform_controller import CUniformController
from controllers.cu_mppi_controller import CUMPPiController
from core.experiment_runner import ExperimentRunner
from utils.visualizer import save_enhanced_visualization
from utils.results_analyzer import ResultsAnalyzer

# Helper function for resume functionality
def scan_completed_trials(base_results_dir, controllers_to_test):
    """Scans temp_trial_results, validates files, finds completed trials, and the last trial count."""
    # Location where ResultsAnalyzer saves intermediate files
    temp_results_dir = os.path.join(base_results_dir, "temp_trial_results")
    completed_trials = set()
    max_trial_count = 0
    
    if not os.path.exists(temp_results_dir):
        return completed_trials, max_trial_count

    # Pattern to extract the count prefix: trial_XXXXX_
    count_pattern = re.compile(r'^trial_(\d+)_')
    
    print(f"Scanning {temp_results_dir} for existing results...")
    for filename in os.listdir(temp_results_dir):
        if not filename.endswith('.pkl'):
            continue

        match = count_pattern.match(filename)
        if not match:
            continue
            
        # 1. Update the maximum trial count found
        try:
            trial_count = int(match.group(1))
            max_trial_count = max(max_trial_count, trial_count)
        except ValueError:
            continue

        # 2. Parse Controller Name and Identifier
        # Extract the filename after the count prefix
        remainder = filename[match.end():]
        ctrl_name = None

        # Iterate through known controllers, sorted by length (longest first)
        sorted_controllers = sorted(controllers_to_test, key=len, reverse=True)
        for name in sorted_controllers:
            prefix = f"{name}_"
            if remainder.startswith(prefix):
                ctrl_name = name
                # The rest is the identifier (remove prefix and .pkl suffix)
                trial_identifier = remainder[len(prefix):-4]
                break
        
        if ctrl_name is None:
            # This might happen if controllers_to_test list changed, but we rely on the loaded config
            print(f"[WARNING] Could not parse controller from filename: {filename}")
            continue

        completed_trials.add((ctrl_name, trial_identifier))
    
    print(f"Found {len(completed_trials)} valid existing results. Initializing ResultsAnalyzer count to: {max_trial_count}.")
    return completed_trials, max_trial_count

class AppConfig:
    """Loads and provides easy access to all experiment configuration settings."""
    def __init__(self, config_path="configs/experiment_config.yaml", resume_dir=None):
        if resume_dir:
            # If resuming, load config from the resume directory's copy
            config_path = os.path.join(resume_dir, 'experiment_config_copy.txt')
            print(f"INFO: Resuming experiment. Loading configuration from: {config_path}")

        if not os.path.exists(config_path):
            # Check existence because we might be loading a specific path now
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            # Store the raw dictionary for passing to other modules
            self.config_dict = yaml.safe_load(f)
        
        # Automatically determine velocity mode based on vrange
        vrange = self.config_dict.get('vrange', [])
        if len(vrange) == 2 and vrange[0] != vrange[1]:
            self.config_dict['variable_velocity_mode'] = True
        else:
            self.config_dict['variable_velocity_mode'] = False
        
        # Dynamically set attributes for easy access, e.g., app_config.max_steps
        for key, value in self.config_dict.items():
            setattr(self, key, value)

def main():
    """Main experiment function."""
    try:
        # --- 0. Argument Parsing ---
        parser = argparse.ArgumentParser(description="Run navigation experiments.")
        parser.add_argument('--num-rollouts', type=int, help="Override the number of rollouts from the config file.")
        parser.add_argument('--resume-dir', type=str, default=None, help="Directory to resume experiment from.")
        args = parser.parse_args()

        # --- 1. Setup and Configuration ---
        print("Starting Navigation Experiments")
        app_config = AppConfig(resume_dir=args.resume_dir)

        completed_trials_set = set()
        starting_trial_count = 0
        if args.resume_dir: # RESUME MODE
            if args.num_rollouts is not None:
                 print(f"WARNING: --num-rollouts ignored when resuming. Using value from config: {app_config.num_rollouts}")
            
            base_results_dir = args.resume_dir
            print(f" Resuming experiment in: {base_results_dir}")

            # Scan for existing results before initializing analyzer and starting the loop
            completed_trials_set, starting_trial_count = scan_completed_trials(base_results_dir, app_config.controllers_to_test)
        else: # NEW EXPERIMENT MODE
            if args.num_rollouts is not None:
                print(f"INFO: Overriding num_rollouts from config with command-line value: {args.num_rollouts}")
                app_config.config_dict['num_rollouts'] = args.num_rollouts
                app_config.num_rollouts = args.num_rollouts
            # Create a unique, timestamped directory (Original behavior)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_type = app_config.experiment_suite['type']
            # Note: Removed the line using trials_per_env here as it's defined later in the original script
            base_results_dir = os.path.join(app_config.results_dir, f"experiment_{timestamp}_{experiment_type}_trajs{app_config.num_rollouts}")
            os.makedirs(base_results_dir, exist_ok=True)
            print(f" Results will be saved to: {base_results_dir}")

            # Save a copy of the experiment configuration (Original behavior)
            config_save_path = os.path.join(base_results_dir, 'experiment_config_copy.txt')
            with open(config_save_path, 'w') as f:
                yaml.dump(app_config.config_dict, f, default_flow_style=False, sort_keys=False)
            print(f"Configuration saved to: {os.path.basename(config_save_path)}")

        print(f"CUDA available: {torch.cuda.is_available()}")
        if app_config.variable_velocity_mode:
            print("INFO: Variable Velocity Mode is ACTIVE (vrange: {}).".format(app_config.vrange))
            # Check if pure C-Uniform controllers are requested
            cuniform_controllers = ["cuniform_map_conditioned", "cuniform_unsupervised_openspace"]
            incompatible = [c for c in app_config.controllers_to_test if c in cuniform_controllers]
            
            if incompatible: # raise error if incompatible controllers are selected
                raise ValueError(
                    f"ERROR: Pure C-Uniform controllers {incompatible} are incompatible with variable velocity mode (vrange[0] != vrange[1]). "
                    "Please use CU-MPPI variants or reconfigure vrange to be fixed."
                )
            
            # check MPPI noise settings
            mppi_controllers_used = any(ctrl.startswith("mppi") or ctrl.startswith("cu_mppi") for ctrl in app_config.controllers_to_test)
            if mppi_controllers_used:
                mppi_config = app_config.config_dict.get('mppi_controller', {})
                u_std = mppi_config.get('u_std', [])
                if len(u_std) >= 1 and u_std[0] == 0.0:
                    print("[WARNING] Variable velocity mode is active and MPPI/CU-MPPI is used, but MPPI velocity noise "
                          f"'u_std[0]' is 0.0. MPPI will not explore the velocity dimension.")

        else:
            print("INFO: FIXED VELOCITY Mode is ACTIVE (vrange: {}).".format(app_config.vrange))

        print(f"CUDA available: {torch.cuda.is_available()}")
        # Create controller-specific subdirectories
        controller_dirs = {}
        for controller_name in app_config.controllers_to_test:
            controller_dir = os.path.join(base_results_dir, controller_name)
            os.makedirs(controller_dir, exist_ok=True)
            controller_dirs[controller_name] = controller_dir

        # --- 2. Initialization ---
        experiment_runner = ExperimentRunner(app_config.config_dict)
        results_analyzer = ResultsAnalyzer(base_results_dir, app_config.config_dict)
        # Set the trial count if resuming to prevent overwriting existing files
        if starting_trial_count > 0:
            results_analyzer.trial_count = starting_trial_count
        environment_manager = EnvironmentsManager(app_config.config_dict)
        environment_manager.set_results_directory(base_results_dir)
        
        # Controller Factory: Maps controller names to their class, config section, and overrides
        CONTROLLER_FACTORY = {
            "mppi": (MPPIPyTorchController, "mppi_controller", {"type_override": 0}),
            "log_mppi": (MPPIPyTorchController, "mppi_controller", {"type_override": 1}),
            "MPPI": (MPPIPyTorchController, "mppi_controller", {"type_override": 0}),
            "LOGMPPI": (MPPIPyTorchController, "mppi_controller", {"type_override": 1}),
            "cuniform_map_conditioned": (CUniformController, "cuniform_controller", {"type_override": 1}),
            "cuniform_unsupervised_openspace": (CUniformController, "cuniform_controller", {"type_override": 0}),
            # Unsupervised C-Uniform + Standard MPPI
            "CU_MPPI": (
                CUMPPiController,
                "cuniform_controller",
                {
                    "type_override": 0, # C-Uniform type: Unsupervised
                    "mppi_type_override": 0, # MPPI type: Standard
                    "mppi_config": app_config.config_dict["mppi_controller"]
                }
            ),
            # Unsupervised C-Uniform + Log-Normal MPPI
            "CU_LOGMPPI": (
                CUMPPiController,
                "cuniform_controller",
                {
                    "type_override": 0, # C-Uniform type: Unsupervised
                    "mppi_type_override": 1, # MPPI type: Log-Normal
                    "mppi_config": app_config.config_dict["mppi_controller"]
                }
            ),
            # Map-Conditioned C-Uniform + Standard MPPI
            "CFU_MPPI": (
                CUMPPiController,
                "cuniform_controller",
                {
                    "type_override": 1, # C-Uniform type: Map-Conditioned
                    "mppi_type_override": 0, # MPPI type: Standard
                    "mppi_config": app_config.config_dict["mppi_controller"]
                }
            ),
            # Map-Conditioned C-Uniform + Log-Normal MPPI
            "CFU_LOGMPPI": (
                CUMPPiController,
                "cuniform_controller",
                {
                    "type_override": 1, # C-Uniform type: Map-Conditioned
                    "mppi_type_override": 1, # MPPI type: Log-Normal
                    "mppi_config": app_config.config_dict["mppi_controller"]
                }
            ),
        }
        
        # --- 3. Experiment Execution ---
        trials_executed_count = 0 # Track executed trials for better resume feedback
        experiment_type = app_config.experiment_suite['type']
        print("\n" + "="*80)
        print("Running Experiments...")
        print(f"Experiment Type: {experiment_type}")
        print(f"Controllers: {app_config.controllers_to_test}")
        print(f"Total test configurations: {len(environment_manager.test_plan)}")
        print("="*80)

        # Main experiment loop - iterate over all environment configurations
        for environment in environment_manager:
            env_info = environment.get_current_env_info()
            config = env_info['config']
            
            print(f"\n--- Environment {env_info['current_index']+1}/{env_info['total_tests']}: " +
                  f"{config['type']} ---")
            
            if config['type'] == 'BARN':
                print(f"    BARN Environment ID: {config['env_id']}")
            elif config['type'] == 'DumbbellSweep':
                print(f"    Dumbbell w={config['width']}, l={config['length']}")
            elif config['type'] == 'BudgetSweep':
                print(f"    BudgetSweep w={config['width']}, l={config['length']}, b={config['sampling_budget']}")
            elif config['type'] == 'Polygon':
                print(f"    Polygon Environment ID: {config['env_id']}")
            
            print(f"    Map shape: {env_info['map_shape']}")

            # Get trial parameters for this specific environment and trial
            start_state, goal_pos = environment.get_trial_params(config['trial_num'])
            if config['type'] == 'Polygon':
                reference_scale_factor = 10
                scale_factor = app_config.config_dict['experiment_suite']['polygon_config']['polygon_scale_factor']
                #NOTE: the start and goal positions are defined w.r.t reference_scale_factor of 10, so we need to scale them down to the actual scale factor
                start_state[:2] /= (reference_scale_factor / scale_factor)
                goal_pos /= (reference_scale_factor / scale_factor)
            
            # need this identifier for saving results and visualization filenames
            if config['type'] == 'BARN':
                env_identifier = f"env{config['env_id']}"
            elif config['type'] == 'DumbbellSweep':
                env_identifier = f"w{config['width']}_l{config['length']}"
            elif config['type'] == 'BudgetSweep':
                env_identifier = f"w{config['width']}_l{config['length']}_b{config['sampling_budget']}"
            elif config['type'] == 'Polygon':
                env_identifier = f"poly_env{config['env_id']}"
            else:
                env_identifier = "unknown_env"
            
            # Create the comprehensive identifier including the trial number
            trial_identifier = f"{env_identifier}_trial{config['trial_num']+1}"
            
            # Test each controller on this environment configuration
            for controller_name in app_config.controllers_to_test:
                # Check against the pre-scanned set of completed (controller, identifier) tuples
                if (controller_name, trial_identifier) in completed_trials_set:
                    print(f"  \n[SKIPPING] Controller: {controller_name}. Valid result already exists for {trial_identifier}.")
                    continue
                print(f"  \nTesting Controller: {controller_name}")

                if controller_name not in CONTROLLER_FACTORY:
                    raise ValueError(f"Unknown controller '{controller_name}' defined in config.")

                # Create a unique, deterministic seed for this specific trial run.
                # env_info['current_index'] is a unique counter for every single test config.
                trial_seed = app_config.seed + env_info['current_index']
                
                ControllerClass, config_section, overrides = CONTROLLER_FACTORY[controller_name]
                controller = ControllerClass(
                    controller_config=app_config.config_dict[config_section],
                    experiment_config=app_config.config_dict,
                    **overrides,
                    seed=trial_seed,
                )
                if 'sampling_budget' in config:
                    new_budget = config['sampling_budget']
                    controller.num_rollouts = new_budget
                    # Handle budget allocation for CU-MPPI controllers
                    if isinstance(controller, CUMPPiController):
                        init_budget = int(new_budget * controller.initialization_budget_ratio)
                        mppi_budget = new_budget - init_budget

                        # Update C-Uniform part
                        controller.num_trajectories_init = init_budget
                        controller.num_trajectories = init_budget

                        # Update MPPI part (the refiner instance)
                        controller.num_trajectories_mppi = mppi_budget
                        controller.mppi_refiner.K = mppi_budget

                        print(f"      > CU-MPPI budget allocation - Total: {new_budget}, Init: {init_budget}, MPPI: {mppi_budget}")
                    elif isinstance(controller, MPPIPyTorchController):
                        controller.K = new_budget
                        print(f"      > MPPIPyTorch budget allocation - Total: {new_budget}")
                    else:
                        # Handle C-Uniform controllers (num_trajectories)
                        if hasattr(controller, 'num_trajectories'):
                            controller.num_trajectories = new_budget
                        if hasattr(controller, 'num_control_rollouts'):
                           controller.num_control_rollouts = new_budget
                    controller.num_vis_rollouts = min(controller.num_vis_rollouts, new_budget)
                    print(f"      > Overriding sampling budget to: {new_budget}")
                
                # Run the trial
                result = experiment_runner.run_trial(controller, environment, start_state, goal_pos)
                trials_executed_count += 1
                
                # Add metadata for analysis and save result
                result['controller'] = controller_name
                result['experiment_type'] = config['type']
                result['trial_num'] = config['trial_num']

                # Add experiment-specific metadata to the result dictionary
                experiment_mapping = {
                    'BARN': {'environment_id': 'env_id'},
                    'DumbbellSweep': {'bottleneck_width': 'width', 'bottleneck_length': 'length'},
                    'BudgetSweep': {'bottleneck_width': 'width', 'bottleneck_length': 'length', 'sampling_budget': 'sampling_budget'},
                    'Polygon': {'environment_id': 'env_id'},
                }
                if config['type'] in experiment_mapping:
                    for key, value in experiment_mapping[config['type']].items():
                        result[key] = config[value]
                
                results_analyzer.save_trial_result(result, controller_name, trial_identifier)
                
                status = "SUCCESS" if result['success'] else result['termination_reason'].upper()
                print(f"      > Result: {status} in {result['steps']} steps")
                
                # Create visualization for this trial
                if len(result['trajectory']) > 0:
                    # Skip animations when maximum steps reached to avoid long processing times
                    skip_animation = result['termination_reason'] == 'max_steps'
                    save_enhanced_visualization(
                        trajectory=result['trajectory'],
                        controls=result['controls'],
                        occupancy_grid=environment.current_map,
                        goal=goal_pos,
                        start=start_state,
                        success=result['success'],
                        termination_reason=result['termination_reason'],
                        sampled_trajectories=result.get('sampled_trajectories', []),
                        costmaps_history=result.get('costmaps_history', []),
                        save_dir=controller_dirs[controller_name],
                        controller_name=controller_name,
                        environment_id=trial_identifier,
                        dt=app_config.dt,
                        experiment_config=app_config.config_dict,
                        visualize_costmap=True,
                        visualize_all_trajectories=True,
                        save_animations=app_config.config_dict['save_animations'] and not skip_animation,
                    )
                    del result
                    del controller
                    # Run Python garbage collection
                    gc.collect()
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # --- 4. Process and Save All Results ---
        if trials_executed_count == 0:
            print("\nINFO: No new trials were executed during this run (all were skipped). Proceeding to analyze existing results.")
        print("\n" + "="*80)
        print("Processing and saving results...")
        results_analyzer.process_and_save()
        print(" Experiment suite completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 