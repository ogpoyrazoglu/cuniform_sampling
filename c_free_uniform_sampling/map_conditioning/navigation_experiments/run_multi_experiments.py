#!/usr/bin/env python3
"""
Multi-experiment automation script with clean progress tracking.
Runs navigation experiments with different num_rollouts values in parallel.
"""
import os
import sys
import subprocess
import time
import yaml
import re
import threading
from datetime import datetime
from collections import defaultdict

class ExperimentMonitor:
    def __init__(self, experiment_id, num_rollouts, total_envs, total_controllers):
        self.experiment_id = experiment_id
        self.num_rollouts = num_rollouts
        self.total_envs = total_envs
        self.total_controllers = total_controllers
        self.completed_envs = 0
        self.current_env = 0
        self.status = "Starting"
        self.start_time = datetime.now()
        self.lock = threading.Lock()
    
    def update_progress(self, line):
        with self.lock:
            # Match environment progress: "--- Environment X/Y: Type ---"
            env_match = re.search(r'Environment\s+(\d+)/(\d+):', line)
            if env_match:
                self.current_env = int(env_match.group(1))
                self.status = f"Env {self.current_env}/{self.total_envs}"
            
            # Match the exact success message from run_experiments.py
            if "Experiment suite completed successfully!" in line:
                self.completed_envs = self.total_envs
                self.status = "Completed"
            # Check for failures
            elif "[ERROR]" in line or "Traceback (most recent call last):" in line:
                 self.status = "Failed"
    
    def get_progress_bar(self, width=20):
        if self.total_envs == 0:
            return "█" * width
        
        progress = self.current_env / self.total_envs
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        return f"{bar} {self.current_env}/{self.total_envs}"
    
    def get_status_line(self):
        elapsed = datetime.now() - self.start_time
        elapsed_str = f"{int(elapsed.total_seconds())}s"
        return f"[{self.experiment_id:2d}] n={self.num_rollouts:3d} │ {self.get_progress_bar(15)} │ {self.status:12s} │ {elapsed_str:>6s}"

def calculate_total_experiments(config_path):
    """Calculate total environments and controllers from config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Count active controllers
    controllers = [c for c in config['controllers_to_test'] if not c.strip().startswith('#')]
    
    # Count environments based on experiment type
    suite = config['experiment_suite']
    experiment_type = suite['type']
    trials_per_env = suite['trials_per_environment']

    num_unique_envs = 0
    if experiment_type == "Polygon":
        num_unique_envs = len(suite['polygon_config']['test_environments'])
    elif experiment_type == "BARN":
        num_unique_envs = len(suite['barn_config']['test_environments'])
    elif experiment_type == "DumbbellSweep":
        widths = len(suite['base_dumbbell_params']['bottleneck_widths'])
        lengths = len(suite['dumbbell_config']['bottleneck_lengths'])
        num_unique_envs = widths * lengths
    elif experiment_type == "BudgetSweep":
        widths = len(suite['base_dumbbell_params']['bottleneck_widths'])
        budgets = len(suite['budget_config']['sampling_budgets_to_test'])
        num_unique_envs = widths * budgets
    else:
        num_unique_envs = 1
    
    total_test_runs = num_unique_envs * trials_per_env
    return total_test_runs, len(controllers)

def monitor_experiment(process, monitor):
    """Monitor experiment output and update progress."""
    try:
        # When using text=True in Popen, we iterate directly over stdout (strings)
        for line in process.stdout:
            monitor.update_progress(line.strip())
    except Exception as e:
        print(f"[Monitor Error] An error occurred in the monitoring thread: {e}")

def run_experiment(experiment_id, num_rollouts_value, total_envs, total_controllers):
    """Run experiment with progress monitoring."""
    monitor = ExperimentMonitor(experiment_id, num_rollouts_value, total_envs, total_controllers)

    command = [
        sys.executable, 
        "-u", # Force unbuffered output
        "run_experiments.py", 
        "--num-rollouts", 
        str(num_rollouts_value)
    ]

    process = subprocess.Popen(command, 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.STDOUT,
                             bufsize=1,
                             text=True)
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_experiment, args=(process, monitor))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    return process, monitor

def display_progress(monitors):
    """Display clean progress for all experiments."""
    print("\033[H\033[J", end="")  # Clear screen and move cursor to top
    
    print(f"┌{'─' * 80}┐")
    print(f"│ Multi-Experiment Progress Monitor {' ' * 37}│")
    print(f"├{'─' * 80}┤")
    print(f"│ ID  Rollouts │ Progress Bar     │ Status       │ Time   │")
    print(f"├{'─' * 80}┤")
    
    for monitor in monitors:
        print(f"│ {monitor.get_status_line()} │")
    
    print(f"└{'─' * 80}┘")
    print(f"\nLast update: {datetime.now().strftime('%H:%M:%S')}")

def main():
    """Main function to run multiple experiments with clean progress tracking."""
    num_rollouts_values = [128, 256, 512, 1024, 2048, 4096]
    
    # Store original directory and change to script directory
    original_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        # Calculate experiment dimensions
        total_envs, total_controllers = calculate_total_experiments("configs/experiment_config.yaml")
        print(f"Each experiment will run {total_envs} environments * {total_controllers} controllers")
        
        processes = []
        monitors = []
        
        # Launch all experiments with delays
        for i, num_rollouts in enumerate(num_rollouts_values):
            if i > 0:
                time.sleep(2)  # 2-second delay between launches to create folder with different time stamp
            
            process, monitor = run_experiment(i+1, num_rollouts, total_envs, total_controllers)
            processes.append(process)
            monitors.append(monitor)
        
        # Clear screen and start monitoring
        print("\033[2J\033[H", end="")  # Clear screen
        
        # Monitor all experiments
        while any(p.poll() is None for p in processes):
            display_progress(monitors)
            time.sleep(1)
        
        # Final update
        display_progress(monitors)
        
        # Show final summary
        successful = sum(1 for p in processes if p.returncode == 0)
        failed = len(processes) - successful
        
        print(f"\n┌{'─' * 50}┐")
        print(f"│ Final Results {' ' * 33}│")
        print(f"├{'─' * 50}┤")
        print(f"│ Total experiments: {len(processes):2d} {' ' * 24}│")
        print(f"│ Successful:        {successful:2d} {' ' * 24}│")
        print(f"│ Failed:            {failed:2d} {' ' * 24}│")
        print(f"└{'─' * 50}┘")
        
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    main()