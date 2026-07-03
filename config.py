"""
Holds constant variables that are used throughout the analysis
"""
from collections import defaultdict
debug = True

condition_preparation_median_times = defaultdict(list)

experiments = [
    r"E:\PID 3",    
    r"E:\PID 4",    
    r"E:\PID 5",    
    r"E:\PID 6",
    r"E:\PID 7",  
    r"E:\PID 8",
    r"E:\PID 9",  
    r"E:\PID 10",  
    r"E:\PID 11",
    r"E:\PID 13", 
    r"E:\PID 14",  
    r"E:\PID 15",
    r"E:\PID 17",
    r"E:\PID 18",  
    r"E:\PID 19",
    r"E:\PID 20",  
    r"E:\PID 22",  
    r"E:\PID 23",  
    r"E:\PID 24",
    r"E:\PID 25",
    r"E:\PID 26",  
    r"E:\PID 27",
    r"E:\PID 28",  
    r"E:\PID 29",  
    r"E:\PID 30",
    r"E:\PID 31",
    r"E:\PID 32",  
    r"E:\PID 33",
    r"E:\PID 35",  
    r"E:\pid 36",  
    r"E:\PID 38",
    r"E:\PID 41",
    r"E:\PID 42",  
    r"E:\PID 43",  
    r"E:\PID 45",  
    r"E:\PID 58",  
    r"E:\PID 46",  
    r"E:\PID 49",  
    r"E:\PID 50",  
    r"E:\PID 52",
    r"E:\PID 55",  
    r"E:\PID 57"  

]  


ica_components = 29
preparation_buffer = 2

pre_crossing_sec = 9
post_crossing_sec = 1.3
eog_threshold = 3.5
longest_crossing_duration = 1.05
fixed_epoch_duration = 7.0
baseline_duration_sec = 2.0

montage_path = r"E:\Montage\Standard-10-10-Cap33_V6.loc"
experiment_root = r"E:\PID 5"
meta_info_path = r"E:\PID 5\UNITY\PID 5 AMBIENT EXPECTED\S005\trial_results.csv"

all_crossing_epochs = []

crossing_durations = 0
crossing_durations_count = 0
#Crossing duration average : 0.5395068106144675

median_prep_csv_path = "median_preparation_durations.csv"

# Saves a number of valid epochs
number_of_valid_epochs = []

all_obs_on_off_markers = []

avg_obstacle_on_time = 0
avg_obstacle_off_time = 0
