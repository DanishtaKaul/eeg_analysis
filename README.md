# Effect of Dim Lighting on Walking and Obstacle Navigation

This pipeline processes mobile EEG recorded during walking and obstacle navigation
and analyses how lighting and obstacle conditions affect neural
activity in young and older adults.

**No participant data is included in this repository**. The scripts read from local
data folders that are not committed. Paths are set at the top of each script and in
`config.py` and `cluster_config.py`.

## Main pipeline

Entry point: `main.py`. It uses `navigation.py` to find each participant's files and
`block_pipeline.py` to run them through the stages below in order. The stages are toggled
inside `main.py` rather than run one at a time.

- `navigation.py` - find each participant's EEG and Unity files and start the pipeline.
- `load_config.py` - load a raw file, rename the channels, and set the montage.
- `block_pipeline.py` - run the full pipeline for one participant.

1. `preprocessing.py` - filter and detrend the EEG, flag bad channels, and run ICA to remove artifacts.
2. `events.py` - group the annotations into trials and label each with its condition.
3. `epoching.py` - segment into per-trial epochs around each obstacle crossing.
4. `autoreject.py` - clean the epochs by interpolating bad channels and rejecting bad epochs.
5. `time_warping.py` - warp the preparation phase to a common duration and remove outliers.
6. `epochs_manual_reject.py` - drop the epochs flagged for manual removal.
7. `trial_counts.py` - count the retained trials per condition and flag low counts.
8. `time_frequency_plots.py` - compute baseline-corrected time-frequency maps at Fz, Cz, and POz.
9. `compute_full_tfrs.py` - compute baseline-corrected time-frequency maps across all channels.
10. `full_tfrs_not_baseline_corrected.py` - compute uncorrected time-frequency maps across all channels.

## Cluster analysis

Run after the pipeline, using the saved time-frequency maps.

- `cluster_config.py` - participant list, age groups, and paths for the cluster analysis.
- `extract_prep_reset.py` - slice the preparation and reset windows out of the saved maps.
- `run_cluster_permutation.py` - test the main effect of age with a cluster permutation test.
- `run_cluster_permutation_light.py` - test the main effect of lighting with a cluster permutation test.
- `run_cluster_permutation_main_effect_obstacle.py` - test the main effect of obstacle with a cluster permutation test.

## Post-hoc tests

Run after the main-effect tests, on the clusters they find.

- `line_plot_cluster_test_light_post_hoc.py` - post-hoc comparisons for main effect of light.
- `line_plot_cluster_test_obstacle_present_absent.py` - compare present and absent obstacle trials.
- `line_plot_cluster_test_obstacle_unexpected_present_expected_present.py` - compare unexpected-present and expected-present trials.
- `line_plot_cluster_test_obstacle_unexpected_absent_expected_absent.py` - compare unexpected-absent and expected-absent trials.
- `line_plot_cluster_test_obstacle_unexpected_present_all_absent.py` - compare unexpected-present trials against absent trials.
- `line_plot_cluster_test_obstacle_expected_present_all_absent.py` - compare expected-present trials against absent trials.

## Supplementary analyses

- `age_light_mixed_anova.py` - test how age and lighting affect regional band power.
- `baseline_frontal_alpha_main_effect_age.py` - measure each participant's frontal baseline alpha.
- `baseline_ttests.py` - run baseline t-tests across lighting and between age groups.
- `calculate_leg_length.py` - estimate each participant's leg length from the motion-tracking data.
- `eeg_height_leg_length_t_test.py` - compare leg length and height between age groups.

## Behavioural Analysis: speed and cadence

Walking speed and cadence computed from the same trials used in the EEG analysis.

Run steps 1 to 5 of the gait pipeline (the `gait_analysis` repository) first to produce the time series these scripts read.

- `eeg_final_dataset_for_gait_analysis_with_obstacle.py` - build the set of EEG-matched gait trials.
- `eeg_calculate_speed_cadence_with_obstacle.py` - compute speed and cadence for each participant and condition.
- `eeg_speed_mixed_model_with_obstacle.R` - fit a mixed model for speed.
- `eeg_cadence_mixed_model_with_obstacle.R` - fit a mixed model for cadence.

## Dependencies

- `config.py` - constants and data paths for the main pipeline.
- `scripts/__init__.py` - shared logger setup.
- `manual_reject_config.py` - the per-participant list of epochs to drop manually.
- `helper_functions.py` - shared helper functions, also used by the gait repository.

## Note

- The lighting condition coded as LIGHT is reported as "Bright" in the paper and figures.

## Author

Danishta Kaul
