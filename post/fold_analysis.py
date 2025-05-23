import os
import json
import glob
import numpy as np
import pandas as pd
from collections import defaultdict


def parse_metrics_json(metrics_data):
    """
    Parse metrics from the JSON file structure.
    Handles both single-target and dual-target formats.
    """
    try:
        # Case 1: Nested structure with a top-level key (e.g., "1")
        if len(metrics_data) == 1 and not any(k.startswith("target_") for k in metrics_data):
            first_key = list(metrics_data.keys())[0]
            inner_data = metrics_data[first_key]

            # Check if inner data has target keys or is directly metrics
            if isinstance(inner_data, dict):
                if "target_0" in inner_data:
                    # Format: {"1": {"target_0": {...}, "target_1": {...}}}
                    return {
                        "target_0": inner_data.get("target_0", {}),
                        "target_1": inner_data.get("target_1", None)
                    }
                elif any(k in inner_data for k in ["mae", "rmse", "r2"]):
                    # Format: {"1": {"mae": 0.123, "rmse": 0.456, "r2": 0.789}}
                    return {
                        "target_0": inner_data,
                        "target_1": None
                    }
                elif len(inner_data) == 2 and all(k.startswith("target_") for k in inner_data):
                    # Format: {"1": {"target_0": {...}, "target_1": {...}}}
                    return inner_data

        # Case 2: Direct target keys
        if "target_0" in metrics_data:
            # Format: {"target_0": {...}, "target_1": {...}}
            return {
                "target_0": metrics_data.get("target_0", {}),
                "target_1": metrics_data.get("target_1", None)
            }

        # Case 3: Direct metrics (assume target_0)
        if any(k in metrics_data for k in ["mae", "rmse", "r2"]):
            # Format: {"mae": 0.123, "rmse": 0.456, "r2": 0.789}
            return {
                "target_0": metrics_data,
                "target_1": None
            }

        # If we get here, the format is unexpected
        print(f"WARNING: Unexpected metrics format: {metrics_data}")
        return {"target_0": {}, "target_1": None}

    except Exception as e:
        print(f"ERROR in parse_metrics_json: {str(e)}")
        print(f"Problematic metrics_data: {metrics_data}")
        return {"target_0": {}, "target_1": None}


def analyze_metrics(base_pattern='XTB_dimenet++_mean_seed*', verbose=True):
    """Analyze metrics from all matching directories."""
    if verbose:
        print(f"Looking for directories matching pattern: {base_pattern}")

    matching_dirs = sorted(glob.glob(base_pattern))

    if not matching_dirs:
        print(f"Error: No directories found matching pattern '{base_pattern}'")
        return [], [], {}

    print(f"Found {len(matching_dirs)} matching directories: {', '.join(matching_dirs)}")

    folder_fold_results = []
    missing_files = []
    all_metrics = defaultdict(list)  # For calculating overall statistics

    for dir_path in matching_dirs:
        if verbose:
            print(f"\nProcessing directory: {dir_path}")

        fold_pattern = os.path.join(dir_path, 'fold_*')
        fold_dirs = sorted(glob.glob(fold_pattern))

        if not fold_dirs:
            print(f"Warning: No fold directories found in {dir_path}")
            continue

        folder_metrics = {
            'folder': dir_path,
            'target_0_mae_mean': [],
            'target_0_mae_std': [],
            'target_0_rmse_mean': [],
            'target_0_rmse_std': [],
            'target_0_r2_mean': [],
            'target_0_r2_std': [],
            'target_1_mae_mean': [],
            'target_1_mae_std': [],
            'target_1_rmse_mean': [],
            'target_1_rmse_std': [],
            'target_1_r2_mean': [],
            'target_1_r2_std': []
        }

        processed_folds = 0
        expected_folds = len(fold_dirs)

        for fold_dir in fold_dirs:
            metrics_path = os.path.join(fold_dir, 'test_metrics.json')

            fold_result = {
                'folder': dir_path,
                'fold': os.path.basename(fold_dir)
            }

            if not os.path.exists(metrics_path):
                missing_files.append(metrics_path)
                print(f"Warning: {metrics_path} not found")

                for target in ['target_0', 'target_1']:
                    for metric in ['mae', 'rmse', 'r2']:
                        fold_result[f'{target}_{metric}'] = np.nan

                folder_fold_results.append(fold_result)
                continue

            try:
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)

                if verbose:
                    print(f"Structure of {metrics_path}:")
                    print(json.dumps(metrics_data, indent=2)[:200] + "..."
                          if len(json.dumps(metrics_data)) > 200
                          else json.dumps(metrics_data, indent=2))

                parsed_metrics = parse_metrics_json(metrics_data)

                for target in ['target_0', 'target_1']:
                    if parsed_metrics[target] is None:
                        continue

                    for metric in ['mae', 'rmse', 'r2']:
                        if metric in parsed_metrics[target]:
                            value = parsed_metrics[target][metric]
                            fold_result[f'{target}_{metric}'] = value
                            folder_metrics[f'{target}_{metric}_mean'].append(value)

                            # Add to all metrics for overall statistics
                            all_metrics[f'{target}_{metric}'].append(value)

                folder_fold_results.append(fold_result)
                processed_folds += 1

            except Exception as e:
                print(f"Error processing {metrics_path}: {str(e)}")
                missing_files.append(metrics_path)

        # Calculate mean and std for each metric in this folder
        for target in ['target_0', 'target_1']:
            for metric in ['mae', 'rmse', 'r2']:
                key_mean = f'{target}_{metric}_mean'
                key_std = f'{target}_{metric}_std'

                values = folder_metrics[key_mean]
                folder_metrics[key_mean] = np.mean(values) if values else np.nan
                folder_metrics[key_std] = np.std(values) if values else np.nan

        folder_fold_results.append(folder_metrics)

        if verbose:
            completion_rate = (processed_folds / expected_folds) * 100
            print(f"Completed {processed_folds}/{expected_folds} folds ({completion_rate:.1f}%)")

    # Create overall statistics for all metrics
    overall_stats = {}
    for key, values in all_metrics.items():
        if values:  # Only calculate if we have values
            overall_stats[f'{key}_mean'] = np.mean(values)
            overall_stats[f'{key}_std'] = np.std(values)
            overall_stats[f'{key}_min'] = np.min(values)
            overall_stats[f'{key}_max'] = np.max(values)
            overall_stats[f'{key}_count'] = len(values)
            overall_stats[f'{key}_missing_rate'] = (1 - len(values) / sum(
                len(glob.glob(os.path.join(d, 'fold_*'))) for d in matching_dirs)) * 100

    # Create DataFrames and export to CSV
    try:
        fold_df = pd.DataFrame(folder_fold_results)

        # Check if 'fold' column exists before filtering
        if 'fold' in fold_df.columns and not fold_df.empty:
            fold_details_df = fold_df[fold_df['fold'].notna()]
            folder_summary_df = fold_df[~fold_df['fold'].notna()]

            fold_details_df.to_csv('fold_metrics_detailed.csv', index=False)
            folder_summary_df.to_csv('folder_metrics_summary.csv', index=False)

            print(f"\nSaved detailed metrics to fold_metrics_detailed.csv ({len(fold_details_df)} rows)")
            print(f"Saved folder summaries to folder_metrics_summary.csv ({len(folder_summary_df)} rows)")
        else:
            if fold_df.empty:
                print("Warning: No results to save to CSV")
            else:
                print("Warning: 'fold' column not found in results dataframe")
                fold_df.to_csv('all_metrics.csv', index=False)
                print(f"Saved all metrics to all_metrics.csv ({len(fold_df)} rows)")
    except Exception as e:
        print(f"Error saving CSV files: {str(e)}")

    return folder_fold_results, missing_files, overall_stats


def print_missing_files_summary(missing_files, matching_dirs):
    """Print a summary of missing files."""
    if not missing_files:
        print("\nNo missing files found.")
        return

    print(f"\nMissing Files ({len(missing_files)}):")
    for file in missing_files:
        print(f"  - {file}")

    # Calculate missing rate by directory
    dir_stats = defaultdict(int)
    for dir_path in matching_dirs:
        fold_count = len(glob.glob(os.path.join(dir_path, 'fold_*')))
        missing_count = sum(1 for f in missing_files if dir_path in f)
        if fold_count > 0:
            dir_stats[dir_path] = {
                'fold_count': fold_count,
                'missing_count': missing_count,
                'missing_rate': (missing_count / fold_count) * 100
            }

    if dir_stats:
        print("\nMissing Files by Directory:")
        for dir_path, stats in dir_stats.items():
            if stats['missing_count'] > 0:
                print(
                    f"  {dir_path}: {stats['missing_count']}/{stats['fold_count']} files missing ({stats['missing_rate']:.1f}%)")


def print_metrics_summary(folder_fold_results, overall_stats):
    """Print a summary of metrics."""
    if not overall_stats:
        print("\nNo metrics to summarize.")
        return

    print("\nOverall Metrics Summary:")
    for target in ['target_0', 'target_1']:
        has_target = any(k.startswith(target) and not np.isnan(v) for k, v in overall_stats.items() if
                         k.endswith('_mean') and k.startswith(target))
        if not has_target:
            continue

        print(f"\n  {target}:")
        for metric in ['mae', 'rmse', 'r2']:
            key = f'{target}_{metric}'
            if f'{key}_mean' in overall_stats:
                mean_val = overall_stats[f'{key}_mean']
                std_val = overall_stats[f'{key}_std']
                min_val = overall_stats[f'{key}_min']
                max_val = overall_stats[f'{key}_max']
                count = overall_stats[f'{key}_count']
                missing_rate = overall_stats.get(f'{key}_missing_rate', 0)

                print(f"    {metric.upper()}:")
                print(f"      Mean ± Std: {mean_val:.4f} ± {std_val:.4f}")
                print(f"      Range: [{min_val:.4f}, {max_val:.4f}]")
                print(f"      Samples: {count} (Missing: {missing_rate:.1f}%)")

    print("\nDetailed Folder Metrics:")
    for summary in folder_fold_results:
        if 'fold' not in summary:
            print(f"\nFolder: {summary['folder']}")
            for target in ['target_0', 'target_1']:
                # Check if any valid metrics exist for this target
                has_values = False
                for metric in ['mae', 'rmse', 'r2']:
                    if f'{target}_{metric}_mean' in summary and not np.isnan(
                            summary.get(f'{target}_{metric}_mean', np.nan)):
                        has_values = True
                        break

                if has_values:
                    print(f"  {target}:")
                    for metric in ['mae', 'rmse', 'r2']:
                        mean_value = summary.get(f'{target}_{metric}_mean', np.nan)
                        std_value = summary.get(f'{target}_{metric}_std', np.nan)

                        # Handle NaN values in formatting
                        if np.isnan(mean_value):
                            mean_str = "NaN"
                        else:
                            mean_str = f"{mean_value:.4f}"

                        if np.isnan(std_value):
                            std_str = "NaN"
                        else:
                            std_str = f"{std_value:.4f}"

                        print(f"    {metric}: Mean={mean_str}, Std={std_str}")


if __name__ == "__main__":
    print("Starting analysis...")

    # Default pattern - can be changed as needed
    pattern = 'XTB_dimenet++_mean_seed*'

    matching_dirs = sorted(glob.glob(pattern))
    if not matching_dirs:
        print(f"No directories found matching pattern '{pattern}'")
        # Try with a more general pattern as fallback
        fallback_pattern = 'XTB_*'
        matching_dirs = sorted(glob.glob(fallback_pattern))
        if matching_dirs:
            print(f"Found {len(matching_dirs)} directories with fallback pattern '{fallback_pattern}'")
            print(f"Available directories: {', '.join(matching_dirs)}")
            pattern = fallback_pattern

    folder_fold_results, missing_files, overall_stats = analyze_metrics(pattern)

    print_missing_files_summary(missing_files, matching_dirs)
    print_metrics_summary(folder_fold_results, overall_stats)

    print("\nAnalysis complete.")