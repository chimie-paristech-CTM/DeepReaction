import optuna
import torch
import os
import sys
from pathlib import Path
import logging
import json
import gc
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from deepreaction import Config, ReactionDataset, ReactionTrainer

# Global variables to cache dataset
global_dataset = None
global_data_splits = None

# Define search space for hyperparameter optimization
SEARCH_SPACE = {
    'num_blocks': [4],
    'cutoff': [5.0],
    'prediction_hidden_layers': [3, 4, 5],
    'prediction_hidden_dim': [128, 256, 512]
}


def calculate_total_combinations():
    """Calculate total number of parameter combinations"""
    total = 1
    for param, values in SEARCH_SPACE.items():
        total *= len(values)
    return total


def validate_search_space():
    """Validate that search space contains all required parameters"""
    required_params = ['num_blocks', 'cutoff', 'prediction_hidden_layers', 'prediction_hidden_dim']
    for param in required_params:
        if param not in SEARCH_SPACE:
            raise ValueError(f"Missing required parameter: {param}")
        if not isinstance(SEARCH_SPACE[param], list) or len(SEARCH_SPACE[param]) == 0:
            raise ValueError(f"Parameter {param} must be a non-empty list")


def prepare_dataset():
    """Load and cache dataset globally to avoid reloading for each trial"""
    global global_dataset, global_data_splits

    if global_dataset is None:
        base_params = {
            # Dataset parameters
            'dataset_root': '../dataset/DATASET_DA_F',
            'dataset_csv': '../dataset/DATASET_DA_F/dataset_xtb_final.csv',
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'target_fields': ['DG_act', 'DrG'],
            'target_weights': [1.0, 1.0],
            'input_features': ['DG_act_xtb', 'DrG_xtb'],
            'file_keywords': ['reactant', 'ts', 'product'],
            'id_field': 'ID',
            'dir_field': 'R_dir',
            'reaction_field': 'smiles',
            'use_scaler': True,
            'readout': 'mean',

            # System parameters
            'random_seed': 42234,
            'num_workers': 4,
            'log_level': 'warning',
            'matmul_precision': 'high',
        }

        config = Config.from_params(base_params)
        global_dataset = ReactionDataset(config=config)
        global_data_splits = global_dataset.get_data_splits()
        print("Dataset loaded and cached globally")


def objective(trial):
    """Objective function for Optuna optimization"""
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()

    # Log trial parameters
    print(f"\nTrial {trial.number} - Search space check:")
    for key, values in SEARCH_SPACE.items():
        print(f"  {key}: {values}")

    # Suggest parameters from search space
    num_blocks = trial.suggest_categorical('num_blocks', SEARCH_SPACE['num_blocks'])
    cutoff = trial.suggest_categorical('cutoff', SEARCH_SPACE['cutoff'])
    pred_layers = trial.suggest_categorical('prediction_hidden_layers', SEARCH_SPACE['prediction_hidden_layers'])
    pred_hidden_dim = trial.suggest_categorical('prediction_hidden_dim', SEARCH_SPACE['prediction_hidden_dim'])

    print(f"Trial {trial.number} - Selected params: num_blocks={num_blocks}, cutoff={cutoff}, "
          f"pred_layers={pred_layers}, pred_hidden_dim={pred_hidden_dim}")

    # Build full parameter dictionary
    params = {
        # Dataset parameters
        'dataset_root': './dataset/DATASET_DA_F',
        'dataset_csv': './dataset/DATASET_DA_F/dataset_xtb_final.csv',
        'target_fields': ['DG_act', 'DrG'],
        'target_weights': [1.0, 1.0],
        'input_features': ['DG_act_xtb', 'DrG_xtb'],
        'file_keywords': ['reactant', 'ts', 'product'],
        'id_field': 'ID',           # Column name for reaction ID
        'dir_field': 'R_dir',       # Column name for directory containing reaction files
        'reaction_field': 'smiles', # Column name for SMILES string representation
        'use_scaler': True,
        'readout': 'mean',

        # Model parameters
        'model_type': 'dimenet++',
        'node_dim': 128,
        'dropout': 0.1,
        'use_layer_norm': False,
        'use_xtb_features': True,
        'max_num_atoms': 100,

        # DimeNet++ specific parameters (with trial suggestions)
        'hidden_channels': 128,
        'num_blocks': num_blocks,  # From trial
        'int_emb_size': 64,
        'basis_emb_size': 8,
        'out_emb_channels': 256,
        'num_spherical': 7,
        'num_radial': 6,
        'cutoff': cutoff,  # From trial
        'envelope_exponent': 5,
        'num_before_skip': 1,
        'num_after_skip': 2,
        'num_output_layers': 3,
        'max_num_neighbors': 32,

        # Readout parameters
        'readout_hidden_dim': 128,
        'readout_num_heads': 4,
        'readout_num_sabs': 2,

        # Prediction head parameters (from trial)
        'prediction_hidden_layers': pred_layers,
        'prediction_hidden_dim': pred_hidden_dim,

        # Training parameters (minimal for hyperparameter search)
        'batch_size': 8,
        'eval_batch_size': 8,
        'lr': 0.0005,
        'max_epochs': 1,  # Quick evaluation
        'min_epochs': 0,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.0001,
        'optimizer': 'adamw',
        'scheduler': 'warmup_cosine',
        'warmup_epochs': 3,
        'min_lr': 1e-7,
        'weight_decay': 0.0001,
        'random_seed': 42234,
        'loss_function': 'mse',
        'gradient_clip_val': 0.0,
        'gradient_accumulation_steps': 1,
        'precision': '32',

        # Output parameters
        'out_dir': f'./results/optuna_trial_{trial.number}',
        'save_best_model': False,  # Don't save models during search
        'save_last_model': False,
        'save_predictions': False,
        'save_interval': 0,

        # System parameters
        'cuda': True,
        'gpu_id': 0,
        'num_workers': 2,
        'strategy': 'auto',
        'num_nodes': 1,
        'devices': 1,
        'log_level': 'error',
        'matmul_precision': 'high',
    }

    trainer = None
    config = None

    try:
        # Create configuration and trainer
        config = Config.from_params(params)
        trainer = ReactionTrainer(config=config)

        # Use cached data splits
        train_data, val_data, test_data, scalers = global_data_splits

        # Train model
        train_metrics = trainer.fit(
            train_dataset=train_data,
            val_dataset=val_data,
            test_dataset=test_data,
            scalers=scalers,
            checkpoint_path=None,
            mode='train'
        )

        # Extract test loss
        test_loss = float('inf')

        if train_metrics and 'test_results' in train_metrics:
            test_results = train_metrics['test_results']
            if isinstance(test_results, dict) and 'test_total_loss' in test_results:
                test_loss = float(test_results['test_total_loss'])
                print(f"Got test_loss: {test_loss}")

        print(f"Final test_loss for trial {trial.number}: {test_loss}")

        # Store trial parameters
        trial.set_user_attr('params', {
            'num_blocks': num_blocks,
            'cutoff': cutoff,
            'prediction_hidden_layers': pred_layers,
            'prediction_hidden_dim': pred_hidden_dim
        })

        return test_loss

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')

    finally:
        # Cleanup to free memory
        try:
            if trainer and hasattr(trainer, 'trainer'):
                del trainer.trainer
            if trainer:
                del trainer
            if config:
                del config
        except:
            pass

        torch.cuda.empty_cache()
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()


def main():
    """Main function to run hyperparameter optimization"""
    # GPU setup
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Using CPU")

    # Validate search space
    validate_search_space()

    # Load dataset once
    prepare_dataset()

    # Calculate total combinations
    total_combinations = calculate_total_combinations()

    # Create Optuna study
    study_name = f"dimenet_hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage_url = f"sqlite:///optuna_studies/{study_name}.db"

    os.makedirs("optuna_studies", exist_ok=True)

    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True
    )

    # Print optimization info
    print(f"\nStarting hyperparameter optimization...")
    print(f"Study name: {study_name}")
    print(f"Search space:")
    for param, values in SEARCH_SPACE.items():
        print(f"  {param}: {values} ({len(values)} options)")
    print(f"Total combinations: {' × '.join(str(len(v)) for v in SEARCH_SPACE.values())} = {total_combinations}")

    # Run optimization
    try:
        study.optimize(objective, n_trials=total_combinations, timeout=None)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")

    # Print results
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETED")
    print("=" * 50)

    print(f"Number of finished trials: {len(study.trials)}")

    if len(study.trials) > 0:
        print(f"Best trial number: {study.best_trial.number}")
        print(f"Best test loss: {study.best_value:.6f}")

        print("\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

    # Save results
    results_file = f"optuna_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results = {
        'study_name': study_name,
        'n_trials': len(study.trials),
        'search_space': SEARCH_SPACE,
        'all_trials': []
    }

    if len(study.trials) > 0:
        results['best_trial_number'] = study.best_trial.number
        results['best_value'] = study.best_value
        results['best_params'] = study.best_params

    for trial in study.trials:
        trial_info = {
            'number': trial.number,
            'value': trial.value if trial.value is not None else 'failed',
            'params': trial.params,
            'state': trial.state.name
        }
        results['all_trials'].append(trial_info)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Study database: {storage_url}")

    # Print summary of all trials
    if len(study.trials) > 0:
        print("\nAll trials summary:")
        for trial in sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf')):
            if trial.value is not None:
                print(f"  Trial {trial.number}: loss={trial.value:.6f}, params={trial.params}")


if __name__ == "__main__":
    main()