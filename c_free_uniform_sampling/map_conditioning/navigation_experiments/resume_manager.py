#!/usr/bin/env python3
"""
Resume Manager for interrupted navigation experiments.
Scans results directory, identifies incomplete experiment batches, and resumes them.
Properly tracks progress and handles batch resumption from temp_trial_results.
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

EXPERIMENT_SCRIPT = "run_experiments.py"
RESULTS_DIR = "results"
CONFIG_PATH = "configs/experiment_config.yaml"

class ExperimentMonitor:
    def __init__(self, experiment_id, num_rollouts, expected_total_trials, resume_dir):
        self.experiment_id = experiment_id
        self.num_rollouts = num_rollouts
        self.expected_total_trials = expected_total_trials
        self.resume_dir = resume_dir
        self.current_completed = 0
        self.status = "Starting"
        self.start_time = datetime.now()
        self.lock = threading.Lock()
        
        # Count existing completed trials at start
        self._count_existing_trials()
    
    def _count_existing_trials(self):
        """Count existing trial results in temp_trial_results directory."""
        temp_dir = os.path.join(self.resume_dir, "temp_trial_results")
        if os.path.exists(temp_dir):
            pkl_files = [f for f in os.listdir(temp_dir) if f.endswith('.pkl')]
            self.current_completed = len(pkl_files)
        else:
            self.current_completed = 0
    
    def update_progress(self, line):
        with self.lock:
            # Look for trial completion indicators - when a trial finishes
            if "Result:" in line and ("SUCCESS" in line or "COLLISION" in line or "MAX_STEPS" in line or "TIMEOUT" in line):
                self.current_completed += 1
                self.status = f"Trial {self.current_completed}/{self.expected_total_trials}"
            
            # Check for environment progress indicators
            env_match = re.search(r'Environment\s+(\d+)/(\d+):', line)
            if env_match:
                current_env = int(env_match.group(1))
                total_envs = int(env_match.group(2))
                self.status = f"Env {current_env}/{total_envs}"
            
            # Check for SKIPPING indicators (when existing results are found)
            if "[SKIPPING]" in line and "Controller:" in line:
                # Don't increment here since these were already counted in _count_existing_trials
                pass
            
            # Check for successful completion
            if "Experiment suite completed successfully!" in line:
                self.status = "Completed"
                self.current_completed = self.expected_total_trials
            
            # Check for errors
            if "[ERROR]" in line:
                self.status = "Error"
            elif "Traceback" in line:
                self.status = "Crashed"
    
    def get_progress_bar(self, width=20):
        if self.expected_total_trials == 0:
            return "█" * width
        
        progress = self.current_completed / self.expected_total_trials
        if progress > 1.0:
            progress = 1.0
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        return f"{bar} {self.current_completed}/{self.expected_total_trials}"
    
    def get_status_line(self):
        elapsed = int((datetime.now() - self.start_time).total_seconds())
        completion_pct = (self.current_completed / self.expected_total_trials * 100) if self.expected_total_trials > 0 else 0
        return f"[{self.experiment_id:2d}] n={self.num_rollouts:4d} │ {self.get_progress_bar(15)} │ {completion_pct:5.1f}% │ {self.status:12s} │ {elapsed:>5d}s"

def calculate_expected_total_trials(config_path):
    """
    Calculate expected total number of trials from configuration.
    
    Total trials = num_environments × trials_per_environment × num_controllers
    For example: 300 environments × 2 trials × 6 controllers = 3600 total trials
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading config {config_path}: {e}")
        return 0, 0
    
    # Count active controllers (exclude commented lines starting with #)
    controllers = [c.strip() for c in config['controllers_to_test'] if not c.strip().startswith('#')]
    
    # Count environments based on experiment type
    suite = config.get('experiment_suite', {})
    experiment_type = suite.get('type')
    trials_per_env = suite.get('trials_per_environment', 1)

    num_unique_envs = 0
    if experiment_type == "Polygon":
        num_unique_envs = len(suite.get('polygon_config', {}).get('test_environments', []))
    elif experiment_type == "BARN":
        num_unique_envs = len(suite.get('barn_config', {}).get('test_environments', []))
    elif experiment_type == "DumbbellSweep":
        widths = len(suite.get('base_dumbbell_params', {}).get('bottleneck_widths', []))
        lengths = len(suite.get('dumbbell_config', {}).get('bottleneck_lengths', []))
        num_unique_envs = widths * lengths
    elif experiment_type == "BudgetSweep":
        widths = len(suite.get('base_dumbbell_params', {}).get('bottleneck_widths', []))
        budgets = len(suite.get('budget_config', {}).get('sampling_budgets_to_test', []))
        num_unique_envs = widths * budgets
    else:
        num_unique_envs = 1
    
    # Total trials = environments × trials_per_env × controllers
    total_trials = num_unique_envs * trials_per_env * len(controllers)
    return total_trials, num_unique_envs

def monitor_experiment_thread(process, monitor):
    """Thread target to monitor experiment output stream."""
    try:
        for line in iter(process.stdout.readline, b''):
            if line:
                monitor.update_progress(line.decode('utf-8', errors='ignore').strip())
    except Exception as e:
        print(f"[Monitor Error] {e}")

def display_progress(monitors):
    """Display clean progress table with detailed trial information."""
    print("\033[H\033[J", end="")  # Clear screen
    print(f"--- Multi-Experiment Resume Manager (Update: {datetime.now().strftime('%H:%M:%S')}) ---")
    print("Progress shows: completed_trials/total_trials (envs × trials_per_env × controllers)")
    print(f"ID  Rollouts │ Progress Bar     │  %%    │ Status       │ Time")
    print("-" * 80)
    for monitor in monitors:
        print(monitor.get_status_line())

def find_latest_experiment_batch(results_dir):
    """Find the latest batch of experiments launched within 60 seconds of each other."""
    pattern = re.compile(r'experiment_(\d{8}_\d{6})_([a-zA-Z]+)_trajs(\d+)')
    found_experiments = []
    
    if not os.path.exists(results_dir):
        return []

    for dirname in os.listdir(results_dir):
        match = pattern.match(dirname)
        if match and os.path.exists(os.path.join(results_dir, dirname, 'experiment_config_copy.txt')):
            timestamp_str, experiment_type, num_rollouts = match.groups()
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                found_experiments.append({
                    'dir': os.path.join(results_dir, dirname),
                    'timestamp': timestamp,
                    'num_rollouts': int(num_rollouts),
                    'basename': dirname
                })
            except ValueError:
                continue
    
    if not found_experiments:
        return []
    
    # Sort by timestamp and group by proximity
    found_experiments.sort(key=lambda x: x['timestamp'], reverse=True)
    latest_time = found_experiments[0]['timestamp']
    batch = [exp for exp in found_experiments if (latest_time - exp['timestamp']).total_seconds() < 60]
    return batch

def check_experiment_completion(experiment_dir):
    """
    Check if experiment is complete by verifying:
    1. comparison_report.txt exists 
    2. Expected number of trials match actual trials in temp_trial_results
    """
    # First check: comparison report exists
    comparison_report = os.path.join(experiment_dir, 'comparison_report.txt')
    if not os.path.exists(comparison_report):
        return False, "No comparison report found"
    
    # Second check: verify trial count matches expected
    config_path = os.path.join(experiment_dir, 'experiment_config_copy.txt')
    if not os.path.exists(config_path):
        return False, "No config copy found"
    
    try:
        expected_trials, _ = calculate_expected_total_trials(config_path)
        
        temp_dir = os.path.join(experiment_dir, "temp_trial_results")
        if os.path.exists(temp_dir):
            actual_trials = len([f for f in os.listdir(temp_dir) if f.endswith('.pkl')])
        else:
            actual_trials = 0
        
        if actual_trials < expected_trials:
            return False, f"Incomplete: {actual_trials}/{expected_trials} trials"
        
        return True, f"Complete: {actual_trials}/{expected_trials} trials"
    
    except Exception as e:
        return False, f"Error checking completion: {e}"

def run_resumable_experiment(experiment_id, exp_info, expected_trials):
    """Launch experiment with resume flag and return process and monitor."""
    monitor = ExperimentMonitor(experiment_id, exp_info['num_rollouts'], expected_trials, exp_info['dir'])
    command = [sys.executable, EXPERIMENT_SCRIPT, "--resume-dir", exp_info['dir']]

    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        bufsize=1,
        universal_newlines=False  # Keep as bytes for proper line reading
    )
    
    # Start monitoring thread
    threading.Thread(target=monitor_experiment_thread, args=(process, monitor), daemon=True).start()
    return process, monitor

def main():
    """Main orchestration function."""
    # Set working directory to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir:
        os.chdir(script_dir)
    
    if not os.path.exists(EXPERIMENT_SCRIPT):
        print(f"Error: Script '{EXPERIMENT_SCRIPT}' not found in {os.getcwd()}")
        return
    
    # Determine results directory
    results_dir = RESULTS_DIR
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
            results_dir = config.get('results_dir', RESULTS_DIR)
    except FileNotFoundError:
        print(f"INFO: {CONFIG_PATH} not found. Using default: '{RESULTS_DIR}'.")
    
    print(f"Searching for latest batch in '{results_dir}'...")
    
    # Find the latest batch of experiments
    latest_batch = find_latest_experiment_batch(results_dir)
    if not latest_batch:
        print("No previous experiment batch found.")
        return
    
    print(f"Found {len(latest_batch)} experiments in latest batch:")
    for exp in latest_batch:
        print(f"  - {exp['basename']}")
    
    # Check completion status for each experiment
    experiments_to_resume = []
    for exp in latest_batch:
        is_complete, status_msg = check_experiment_completion(exp['dir'])
        if is_complete:
            print(f"  [SKIPPING] {exp['basename']}: {status_msg}")
        else:
            print(f"  [RESUMING] {exp['basename']}: {status_msg}")
            experiments_to_resume.append(exp)
    
    if not experiments_to_resume:
        print("\nAll experiments in the latest batch are complete.")
        return
    
    print(f"\nResuming {len(experiments_to_resume)} incomplete experiments...")
    
    # Calculate expected trials for monitoring (use first experiment config as reference)
    ref_config = os.path.join(experiments_to_resume[0]['dir'], 'experiment_config_copy.txt')
    expected_trials, num_envs = calculate_expected_total_trials(ref_config)
    
    if expected_trials == 0:
        print("Error: Could not determine expected trial count.")
        return
    
    # Get additional details for better explanation
    try:
        with open(ref_config, 'r') as f:
            config = yaml.safe_load(f)
        controllers = [c.strip() for c in config['controllers_to_test'] if not c.strip().startswith('#')]
        trials_per_env = config['experiment_suite']['trials_per_environment']
        print(f"Expected trials calculation: {num_envs} environments × {trials_per_env} trials_per_env × {len(controllers)} controllers = {expected_trials} total trials")
    except:
        print(f"Each experiment should have {expected_trials} total trials when complete.")
    
    print("Starting resume processes...\n")
    
    # Launch all resume processes
    processes = []
    monitors = []
    
    for i, exp in enumerate(experiments_to_resume):
        process, monitor = run_resumable_experiment(i+1, exp, expected_trials)
        processes.append(process)
        monitors.append(monitor)
        time.sleep(1)  # Stagger launches slightly
    
    # Monitor all processes until completion
    try:
        while any(p.poll() is None for p in processes):
            display_progress(monitors)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Terminating processes...")
        for p in processes:
            if p.poll() is None:
                p.terminate()
        return
    
    # Final status display
    display_progress(monitors)
    
    # Final summary
    successful = sum(1 for p in processes if p.returncode == 0)
    failed = len(processes) - successful
    
    print(f"\n--- Resume Summary ---")
    print(f"Total resumed: {len(processes)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed experiments:")
        for i, p in enumerate(processes):
            if p.returncode != 0:
                print(f"  - {experiments_to_resume[i]['basename']} (exit code: {p.returncode})")

if __name__ == "__main__":
    main()