import os
import json
import glob
import numpy as np
import pandas as pd


def parse_metrics_json(metrics_data):

    first_key = list(metrics_data.keys())[0]
    target_data = metrics_data[first_key]


    if len(target_data) == 1:
        return {
            "target_0": target_data.get("target_0", {}),
            "target_1": None
        }

    return target_data


def analyze_metrics():

    base_pattern = 'hyperopt_cut*_blk*_phl*_phd*_*_*'
    matching_dirs = sorted(glob.glob(base_pattern))


    folder_fold_results = []

    missing_files = []


    for dir_path in matching_dirs:

        fold_pattern = os.path.join(dir_path, 'fold_*')
        fold_dirs = sorted(glob.glob(fold_pattern))


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

            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)

            parsed_metrics = parse_metrics_json(metrics_data)


            for target in ['target_0', 'target_1']:

                if parsed_metrics[target] is None:
                    continue

                for metric in ['mae', 'rmse', 'r2']:
                    if metric in parsed_metrics[target]:
                        value = parsed_metrics[target][metric]


                        fold_result[f'{target}_{metric}'] = value


                        folder_metrics[f'{target}_{metric}_mean'].append(value)


            folder_fold_results.append(fold_result)


        for target in ['target_0', 'target_1']:
            for metric in ['mae', 'rmse', 'r2']:
                key_mean = f'{target}_{metric}_mean'
                key_std = f'{target}_{metric}_std'

                values = folder_metrics[key_mean]
                folder_metrics[key_mean] = np.mean(values) if values else np.nan
                folder_metrics[key_std] = np.std(values) if values else np.nan


        folder_fold_results.append(folder_metrics)


    fold_df = pd.DataFrame(folder_fold_results)


    fold_details_df = fold_df[fold_df['fold'].notna()]
    folder_summary_df = fold_df[~fold_df['fold'].notna()]


    fold_details_df.to_csv('fold_metrics_detailed.csv', index=False)


    folder_summary_df.to_csv('folder_metrics_summary.csv', index=False)

    print("\nMissing Files:")
    for file in missing_files:
        print(file)

    return folder_fold_results, missing_files


folder_fold_results, missing_files = analyze_metrics()


print("\nDetailed Folder Metrics:")
for summary in folder_fold_results:
    if 'fold' not in summary:
        print(f"\nFolder: {summary['folder']}")
        for target in ['target_0', 'target_1']:
            print(f"  {target}:")
            for metric in ['mae', 'rmse', 'r2']:
                print(
                    f"    {metric}: Mean={summary[f'{target}_{metric}_mean']:.4f}, Std={summary[f'{target}_{metric}_std']:.4f}")