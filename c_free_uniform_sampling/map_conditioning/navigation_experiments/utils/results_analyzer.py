import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

class ResultsAnalyzer:
    """
    Collects, processes, and saves all experiment results.
    Assumes all required keys exist in the result dictionaries, will raise KeyError if assumption is violated.
    """
    def __init__(self, base_results_dir: str, experiment_config: Dict[str, Any]):
        self.base_results_dir = base_results_dir
        self.config = experiment_config
        self.temp_results_dir = os.path.join(self.base_results_dir, "temp_trial_results")
        os.makedirs(self.temp_results_dir, exist_ok=True)
        self.trial_count = 0
    
    def save_trial_result(self, result: Dict[str, Any], controller_name: str, trial_identifier: str):
        """Saves a single trial result immediately to a pickle file."""
        self.trial_count += 1
        
        # Define a unique filename for the temporary pickle file
        filename = f"trial_{self.trial_count:05d}_{controller_name}_{trial_identifier}.pkl"
        filepath = os.path.join(self.temp_results_dir, filename)

        # Create a copy before modification to preserve the original result for visualization
        result_to_save = result.copy() 
        if not self.config['save_full_history']:
            result_to_save.pop('costmaps_history', None)
            result_to_save.pop('sampled_trajectories', None)

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(result_to_save, f)
        except Exception as e:
            print(f"[ERROR] Failed to save trial result to {filepath}: {e}")
    
    def _load_all_results(self) -> List[Dict[str, Any]]:
        """Loads all temporary trial results from the disk."""
        all_results = []
        print(f"Loading {self.trial_count} trial results from disk...")
        # Sort ensures trials are loaded in the order they were executed
        for filename in sorted(os.listdir(self.temp_results_dir)):
            if filename.endswith(".pkl"):
                filepath = os.path.join(self.temp_results_dir, filename)
                try:
                    with open(filepath, 'rb') as f:
                        result = pickle.load(f)
                        all_results.append(result)
                except Exception as e:
                    print(f"[ERROR] Failed to load trial result from {filepath}: {e}")
        return all_results

    def process_and_save(self):
        """Processes all collected results, saves them, and prints a summary."""
        if self.trial_count == 0:
            print("[Warning] No results were added to the analyzer.")
            return
        
        all_results = self._load_all_results()
        if not all_results:
             print("[Warning] Failed to load results from disk.")
             return

        results_by_controller = self._group_by_controller(all_results)
        for name, results in results_by_controller.items():
            controller_dir = os.path.join(self.base_results_dir, name)
            self._save_combined_controller_pkl(name, results, controller_dir)

        # Save combined comparison report
        self._save_combined_comparison_report(results_by_controller)

        # Generate heatmap analysis for DumbbellSweep and BudgetSweep experiments
        if self.config['experiment_suite']['type'] == 'DumbbellSweep':
            print("\nGenerating performance heatmaps for DumbbellSweep...")
            self._create_length_vs_width_heatmap(results_by_controller)
        elif self.config['experiment_suite']['type'] == 'BudgetSweep':
            print("\nGenerating performance heatmaps for BudgetSweep...")
            self._create_budget_vs_width_heatmap(results_by_controller)
        
        self._print_final_summary(results_by_controller)
        print(f"\n All results processed and saved in: {self.base_results_dir}")

    def _group_by_controller(self, all_results: List[Dict]) -> Dict[str, List[Dict]]:
        """Groups results by controller name. Assumes 'controller' key exists."""
        grouped = {}
        for r in all_results:
            name = r['controller']
            if name not in grouped:
                grouped[name] = []
            grouped[name].append(r)
        return grouped
    
    def _save_combined_controller_pkl(self, name: str, results: List[Dict], save_dir: str):
        """Saves a combined pickle file containing all trials for a specific controller."""
        filepath = os.path.join(save_dir, f"{name}_combined_results.pkl")
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
            print(f"  > Saved combined data for '{name}' to: {os.path.basename(filepath)}")
        except Exception as e:
             print(f"[ERROR] Failed to save combined controller PKL {filepath}: {e}")

    def _calculate_summary_stats(self, results: List[Dict]) -> Dict[str, float]:
        """Calculates summary statistics for a list of results."""
        successful = [r for r in results if r['success']]
        if not results:
            return {'success_rate': 0, 'avg_steps': 0, 'avg_distance': 0, 'avg_smoothness': 0}

        stats = {'success_rate': len(successful) / len(results) * 100}
        if successful:
            stats['avg_steps'] = np.mean([r['steps'] for r in successful])
            stats['avg_distance'] = np.mean([r['traveled_distance'] for r in successful])
            stats['avg_smoothness'] = np.mean([r['control_smoothness'] for r in successful])
        else: # Handle case with zero successful trials
            stats.update({'avg_steps': 0, 'avg_distance': 0, 'avg_smoothness': 0})
        return stats

    def _save_combined_comparison_report(self, results_by_controller: Dict[str, List]):
        """Saves a combined comparison report with dynamic column alignment."""
        filepath = os.path.join(self.base_results_dir, "comparison_report.txt")
        
        # Find the length of the longest controller name to set the column width.
        controller_names = list(results_by_controller.keys())
        max_name_len = max(len(name) for name in controller_names) if controller_names else 0
        col_width = max(max_name_len, len("Controller")) + 4  # Add 4 for padding

        # Group all results by base environment ID
        unique_env_ids = []
        results_by_env = {} # {env_id: [list_of_all_results_for_that_env]}
        for controller_name, results in results_by_controller.items():
            for result in results:
                # Create a unique identifier for each environment configuration
                if result['experiment_type'] == 'BARN':
                    env_id = f"BARN_{result['environment_id']}"
                elif result['experiment_type'] == 'DumbbellSweep':
                    env_id = f"Dumbbell_w{result['bottleneck_width']}_l{result['bottleneck_length']}"
                elif result['experiment_type'] == 'BudgetSweep':
                    env_id = f"Budget_w{result['bottleneck_width']}_l{result['bottleneck_length']}_b{result['sampling_budget']}"
                elif result['experiment_type'] == 'Polygon':
                    env_id = f"Polygon_{result['environment_id']}"

                if env_id not in unique_env_ids:
                    unique_env_ids.append(env_id)
                if env_id not in results_by_env:
                    results_by_env[env_id] = []
                results_by_env[env_id].append(result)
        
        with open(filepath, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("EXPERIMENT COMPARISON REPORT\n")
            f.write("=" * 100 + "\n\n")
            
            # Section 1: Per-environment comparison
            f.write("SECTION 1: PER-ENVIRONMENT COMPARISON\n")
            f.write("=" * 100 + "\n\n")

            # Find the length of the longest controller name for column alignment.
            controller_names = list(results_by_controller.keys())
            max_name_len = max(len(name) for name in controller_names) if controller_names else 0
            col_width = max(max_name_len, len("Controller")) + 4
            # Iterate through each environment group
            for env_id, trials in sorted(results_by_env.items()):
                f.write(f"Environment: {env_id}\n")
                
                # Note the new "Trial" column in the header
                header = (f"{'Controller':<{col_width}}{'Trial':<7}{'Success':<10}{'Steps':<8}"
                        f"{'Distance':<12}{'Smoothness':<12}{'Termination':<15}")
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
                
                # Sort trials by trial number and then controller name for clean output and apple to apple comparison
                sorted_trials = sorted(trials, key=lambda r: (r['trial_num'], r['controller']))
                
                # Print a line for every single trial
                for result in sorted_trials:
                    success_str = "YES" if result['success'] else "NO"
                    termination = '---' if result['success'] else result['termination_reason']
                    
                    f.write(f"{result['controller']:<{col_width}}"
                            f"{result['trial_num'] + 1:<7}"
                            f"{success_str:<10}"
                            f"{result['steps']:<8}"
                            f"{result['traveled_distance']:<12.2f}"
                            f"{result['control_smoothness']:<12.3f}"
                            f"{termination:<15}\n")
                f.write("\n")
            
            # Section 2: Controller summary comparison
            f.write("SECTION 2: CONTROLLER SUMMARY COMPARISON\n")
            f.write("=" * 100 + "\n\n")
            
            summary_header = (f"{'Controller':<{col_width}}{'Success Rate':<15}{'Avg Steps':<12}"
                            f"{'Avg Distance':<15}{'Avg Smoothness':<15}")
            f.write(summary_header + "\n")
            f.write("-" * len(summary_header) + "\n")
            
            for controller_name in controller_names:
                stats = self._calculate_summary_stats(results_by_controller[controller_name])
                success_rate_str = f"{stats['success_rate']:.1f}%"
                f.write(f"{controller_name:<{col_width}}"
                        f"{success_rate_str:<15}"
                        f"{stats['avg_steps']:<12.1f}"
                        f"{stats['avg_distance']:<15.2f}"
                        f"{stats['avg_smoothness']:<15.3f}\n")
                
            f.write("\n" + "=" * 100 + "\n")
            f.write("NOTE: Averages are calculated only from successful trials.\n")
            f.write("=" * 100 + "\n")
        
        print(f"  > Saved comparison report to: {os.path.basename(filepath)}")


    def _print_final_summary(self, results_by_controller: Dict[str, List]):
        """Prints the final summary to the console."""
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        for name, results in results_by_controller.items():
            stats = self._calculate_summary_stats(results)
            print(f"\n--- Controller: {name} ---")
            success_rate_str = f"{stats['success_rate']:.1f}%"
            print(f"  Success Rate: {success_rate_str} ({len([r for r in results if r['success']])}/{len(results)})")
            if stats['avg_steps'] > 0:
                print(f"  Avg Steps (on success): {stats['avg_steps']:.1f}")
                print(f"  Avg Traveled Distance (on success): {stats['avg_distance']:.2f}m")
                print(f"  Avg Control Smoothness (on success): {stats['avg_smoothness']:.3f}")

    def _create_length_vs_width_heatmap(self, results_by_controller: Dict[str, List[Dict]]):
        """
        Generates and saves a 2D heatmap of success rates for each controller
        in a DumbbellSweep experiment.

        The heatmap shows performance across different bottleneck widths and lengths.
        """
        # Loop through each controller and generate a separate heatmap
        for controller_name, results in results_by_controller.items():
            
            # --- 1. Data Wrangling: Aggregate success rates ---
            # Filter for only DumbbellSweep results for this controller
            dumbbell_results = [r for r in results if r['experiment_type'] == 'DumbbellSweep']
            if not dumbbell_results:
                continue # Skip if this controller had no dumbbell trials

            # Create a data structure to hold success info for each (width, length) pair
            # e.g., {(10, 5): [True, False, True], (10, 10): [True, True, True]}
            performance_data = {}
            for r in dumbbell_results:
                key = (r['bottleneck_width'], r['bottleneck_length'])
                if key not in performance_data:
                    performance_data[key] = []
                performance_data[key].append(r['success'])

            # Calculate the success rate for each pair
            success_rates = {key: np.mean(val) for key, val in performance_data.items()}

            # --- 2. Matrix Creation: Prepare data for the heatmap plot ---
            if not success_rates:
                continue

            # Get the unique, sorted widths and lengths to use as axis labels
            widths = sorted(list(set(k[0] for k in success_rates.keys())))
            lengths = sorted(list(set(k[1] for k in success_rates.keys())))
            
            # Create an empty matrix to hold the success rates
            heatmap_matrix = np.zeros((len(lengths), len(widths)))

            # Populate the matrix with success rates
            for i, length in enumerate(lengths):
                for j, width in enumerate(widths):
                    heatmap_matrix[i, j] = success_rates.get((width, length), 0.0) # Default to 0 if no data

            # --- 3. Plotting: Generate and save the heatmap ---
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                heatmap_matrix,
                annot=True,          # Annotate each cell with its value
                fmt=".0%",           # Format annotations as percentages (e.g., "80%")
                cmap="viridis",      # Use a color-blind friendly colormap
                vmin=0.0,            # sets the minimum of the colormap to 0%(success rate)
                vmax=1.0,            # sets the maximum of the colormap to 100%
                xticklabels=widths,
                yticklabels=lengths,
                ax=ax,
                linewidths=.5,
                cbar_kws={'label': 'Success Rate'} # Label for the color bar
            )
            
            ax.set_title(f"Success Rate Heatmap: {controller_name}", fontsize=16, fontweight='bold')
            ax.set_xlabel("Bottleneck Width (grid cells)", fontsize=12)
            ax.set_ylabel("Bottleneck Length (grid cells)", fontsize=12)
            ax.invert_yaxis()

            # Save the figure
            save_path = os.path.join(self.base_results_dir, f"success_rate_heatmap_{controller_name}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  > Saved heatmap for '{controller_name}' to: {os.path.basename(save_path)}")
    
    def _create_budget_vs_width_heatmap(self, results_by_controller: Dict[str, List[Dict]]):
        """ Generates a 2D heatmap of success rates vs. bottleneck width and sampling budget.  """
        for controller_name, results in results_by_controller.items():
            budget_results = [r for r in results if r.get('experiment_type') == 'BudgetSweep']
            if not budget_results: continue

            # group by width and budget, then calculate the mean success
            performance_data = pd.DataFrame(budget_results).groupby(
                ['bottleneck_width', 'sampling_budget']
            )['success'].mean()
            
            # Pivot the data into a matrix format suitable for a heatmap
            heatmap_matrix = performance_data.unstack(level='bottleneck_width')

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                heatmap_matrix,
                annot=True, fmt=".0%", cmap="viridis", vmin=0.0, vmax=1.0,
                ax=ax, linewidths=.5, cbar_kws={'label': 'Success Rate'}
            )
            
            ax.set_title(f"Success Rate vs. Sampling Budget: {controller_name}", fontsize=16, fontweight='bold')
            ax.set_xlabel("Bottleneck Width (grid cells)", fontsize=12)
            ax.set_ylabel("Sampling Budget (num_rollouts)", fontsize=12)

            save_path = os.path.join(self.base_results_dir, f"success_rate_heatmap_{controller_name}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  > Saved budget heatmap for '{controller_name}' to: {os.path.basename(save_path)}")