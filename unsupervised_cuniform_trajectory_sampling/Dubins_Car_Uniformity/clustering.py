import math
# from pprint import pprint
import pickle
import numpy as np
from classes.grid import Grid

print("Reading pickles...")
# file_name = "reach_data_v_2.0seed_-1_grid_0.10_0.10_9.00deg_na5_t2.01_ts0.1.pkl" 
file_name = "FINAL_reach_data_v_1.0seed_2025_grid_0.05_0.05_1.00deg_na31_t2.01_ts0.2.pkl"
with open(file_name, "rb") as f:
    data = pickle.load(f)
print("Pickles reading done!")

state_u_distribution_across_LS = data["state_u_distribution_across_LS"] # type list

print("Before lower resolution discretization...")
for n in range(len(state_u_distribution_across_LS)): # seems like the last level set is not saved here
    print(f"    level set {n} has {len(state_u_distribution_across_LS[n])} cells")

#Each index of state_u_... has something like the following, given a state, the corresponding action probability distribution
''' 
[([0.2, 0.0, 0.15707963267948966], array([0.06666667, 0.06666667, 0.06666667, 0.4       , 0.4       ],
      dtype=float32)), ([0.2, 0.0, 0.0], array([0.11111111, 0.11111111, 0.2777778 , 0.33333334, 0.16666667],
      dtype=float32)), ([0.2, -0.1, 0.0], array([0.2777778 , 0.2777778 , 0.11111111, 0.16666667, 0.16666667],
      dtype=float32)), ([0.2, -0.1, -0.15707963267948966], array([0.4       , 0.4       , 0.06666667, 0.06666667, 0.06666667],
      dtype=float32)), ([0.2, 0.0, -0.15707963267948966], array([0.23333333, 0.23333333, 0.06666667, 0.23333333, 0.23333333],
      dtype=float32))]
'''

# original_threshold = [0.10, 0.10, (2*math.pi)/40]
original_threshold = data['grid'].thresholds
growth_factor = 1.2 #NOTE: this is an important hyperparameter
increased_threshold = np.array(original_threshold) * growth_factor
new_st_with_lower_resolution = []

print("Processing with larger cell size to decrease sample size for more efficient level set representation")
for n in range(len(state_u_distribution_across_LS)):
    level_set = state_u_distribution_across_LS[n]
    states = np.array([entry[0] for entry in level_set])  # first element in each tuple is the state
    states_inds = set()

    if len(state_u_distribution_across_LS[n]) > 1000:
        g = Grid(thresholds=increased_threshold)
        increased_threshold *= growth_factor
        for st in states:
            st_ind = g.get_index(st)
            states_inds.add(st_ind)
        states = []
        for st_ind in states_inds:
            states.append(g.get_grid_center(st_ind)) 
    new_st_with_lower_resolution.append(states)

print("After lower resolution discretization...")
for n in range(len(new_st_with_lower_resolution)): # seems like the last level set is not saved here
    print(f"    level set {n} has {len(new_st_with_lower_resolution[n])} cells")

with open(f"pruning_gf_{growth_factor}_{file_name}", 'wb') as f:
    pickle.dump(new_st_with_lower_resolution, f)

# print("new_st_with_lower_resolution[11]")
# pprint(new_st_with_lower_resolution[11])
# print("u_states...[11]")
# pprint(state_u_distribution_across_LS[11][:50])