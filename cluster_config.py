# -*- coding: utf-8 -*-
"""Config for the cluster analysis: participant list, Young/Old groups, and TFR/epoch paths"""


participants = [
    "PID 3", "PID 4", "PID 5", "PID 6", "PID 7", "PID 8", "PID 9", "PID 10", "PID 11", "PID 13",
    "PID 14", "PID 15", "PID 17", "PID 18", "PID 19", "PID 20", "PID 22", "PID 23",
    "PID 24", "PID 25", "PID 26", "PID 27", "PID 28", "PID 29", "PID 30", "PID 31", "PID 32", "PID 33", "PID 35",
    "PID 36", "PID 38", "PID 41", "PID 42", "PID 43", "PID 45", "PID 58", "PID 46", "PID 49", "PID 50",
    "PID 52", "PID 55", "PID 57"
]

Young = {"PID 3", "PID 4", "PID 5", "PID 6", "PID 7", "PID 8", "PID 9", "PID 10", "PID 11", "PID 14", "PID 15",
         "PID 18", "PID 19", "PID 20", "PID 23", "PID 26", "PID 30", "PID 31", "PID 35", "PID 50", "PID 57", "PID 58"}

Old = {"PID 13", "PID 17", "PID 22", "PID 24", "PID 25", "PID 27", "PID 28", "PID 29", "PID 32", "PID 33",
       "PID 36", "PID 38", "PID 41", "PID 42", "PID 43", "PID 45", "PID 46", "PID 49", "PID 52", "PID 55"}

base_dir = r"D:\tfr_full"


# uncorrected TFRs (in dB) path
raw_tfr_dir = r"D:\tfr_full_rawdb"

aligned_epochs_dir = r"D:\aligned_epochs"
adjacency_save_dir = r"D:\Adjacency_matrix"

# use any existing epochs file for channel layout
sample_pid = "PID 3"
sample_condition = "GLOBAL_LIGHT_UNEXPECTED_PRESENT"
