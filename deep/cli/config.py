import argparse
import json
import os
import yaml
import logging
from pathlib import Path
from datetime import datetime

SUPPORTED_DATASETS = ['XTB']
SUPPORTED_READOUTS = ['set_transformer', 'mean', 'sum', 'max', 'attention', 'multihead_attention', 'set2set',
                      'sort_pool']
SUPPORTED_OPTIMIZERS = ['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad']
SUPPORTED_LR_SCHEDULERS = ['cosine', 'step', 'exponential', 'plateau', 'warmup_cosine', 'cyclic', 'one_cycle',
                           'constant', 'warmup_constant']
SUPPORTED_LOSS_FUNCTIONS = ['mse', 'mae', 'huber', 'smooth_l1', 'cross_entropy', 'binary_cross_entropy',
                            'evidence_lower_bound']
SUPPORTED_MODELS = ['dimenet++', 'schnet', 'egnn']
MAX_NUM_ATOMS_IN_MOL = {'XTB': 100}

DEFAULT_REACTION_DATASET_ROOT = '/DATASET_DA'
DEFAULT_REACTION_DATASET_CSV = '/DATASET_DA/DA_dataset_cleaned.csv'


def get_model_name(args):
    """Create a descriptive model name based on configuration parameters."""
    components = []

    # Add primary identifiers
    components.append(args.dataset)
    components.append(args.model_type)
    components.append(args.readout)

    # Add important hyperparameters
    components.append(f"blk{args.num_blocks}")
    components.append(f"cut{args.cutoff}")
    components.append(f"hid{args.hidden_channels}")
    components.append(f"lr{args.lr}")
    components.append(f"bs{args.batch_size}")
    components.append(f"seed{args.random_seed}")

    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    components.append(timestamp)

    # Join with underscores
    return "_".join(components)


def print_args_summary(args):
    """Print a summary of the most important arguments for logging purposes."""
    summary = [
        f"Dataset: {args.dataset}",
        f"Model type: {args.model_type}",
        f"Readout: {args.readout}",
        f"Num blocks: {args.num_blocks}",
        f"Cutoff: {args.cutoff}",
        f"Hidden channels: {args.hidden_channels}",
        f"Batch size: {args.batch_size}",
        f"Learning rate: {args.lr}",
        f"Optimizer: {args.optimizer}",
        f"Scheduler: {args.scheduler}",
        f"Random seed: {args.random_seed}",
        f"Max epochs: {args.max_epochs}",
        f"Output directory: {args.output_dir}"
    ]

    return "\n".join(summary)


def get_parser():
    parser = argparse.ArgumentParser(description='Train a molecular graph neural network for property prediction')

    general_group = parser.add_argument_group('General Parameters')
    general_group.add_argument('--config', type=str, default=None, help='Path to a YAML or JSON configuration file')
    general_group.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    general_group.add_argument('--cuda', default=True, action=argparse.BooleanOptionalAction,
                               help='Use CUDA for training if available')
    general_group.add_argument('--precision', type=str, default='32', choices=['16', '32', 'bf16', 'mixed'],
                               help='Floating point precision')
    general_group.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')

    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument('--dataset', type=str, required=True, choices=SUPPORTED_DATASETS, help='Dataset to use')
    data_group.add_argument('--split_type', type=str, default='random',
                            choices=['random', 'scaffold', 'stratified', 'temporal'])
    data_group.add_argument('--train_ratio', type=float, default=0.8,
                            help='Ratio of training data when using automatic splitting')
    data_group.add_argument('--val_ratio', type=float, default=0.1,
                            help='Ratio of validation data when using automatic splitting')
    data_group.add_argument('--test_ratio', type=float, default=0.1,
                            help='Ratio of test data when using automatic splitting')
    data_group.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    data_group.add_argument('--use_scaler', action='store_true', default=True, help='Scale targets')
    data_group.add_argument('--dataset_csv', type=str, default=None,
                            help='Path to dataset CSV file (used as training set if val_csv and test_csv are provided)')
    data_group.add_argument('--val_csv', type=str, default=None, help='Path to validation set CSV (optional)')
    data_group.add_argument('--test_csv', type=str, default=None, help='Path to test set CSV (optional)')

    # Cross-validation parameters
    data_group.add_argument('--cv_folds', type=int, default=0,
                            help='Number of folds for cross-validation (0 to disable)')
    data_group.add_argument('--cv_test_fold', type=int, default=-1,
                            help='Fold to use for testing (-1 means use a fraction of each fold)')
    data_group.add_argument('--cv_stratify', action='store_true', default=False,
                            help='Stratify folds based on target values')
    data_group.add_argument('--cv_grouped', action='store_true', default=True,
                            help='Keep molecules with the same reaction_id in the same fold')

    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--model_type', type=str, default='dimenet++', choices=SUPPORTED_MODELS,
                             help='Type of molecular model to use')
    model_group.add_argument('--readout', type=str, required=True, choices=SUPPORTED_READOUTS, help='Readout function')
    model_group.add_argument('--node_latent_dim', type=int, default=128, help='Node latent dimension')
    model_group.add_argument('--edge_latent_dim', type=int, default=64, help='Edge latent dimension')
    model_group.add_argument('--use_layer_norm', action='store_true', default=False, help='Use layer normalization')
    model_group.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    model_group.add_argument('--activation', type=str, default='silu',
                             choices=['relu', 'leaky_relu', 'elu', 'gelu', 'silu', 'swish'])
    model_group.add_argument('--use_xtb_features', action='store_true', default=True, help='Use XTB features')
    model_group.add_argument('--prediction_hidden_layers', type=int, default=3,
                             help='Number of hidden layers in prediction MLP')
    model_group.add_argument('--prediction_hidden_dim', type=int, default=128,
                             help='Hidden dimension for all layers in prediction MLP')

    model_params_group = parser.add_argument_group('Model Parameters')
    model_params_group.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels in the model')
    model_params_group.add_argument('--num_blocks', type=int, default=4, help='Number of interaction blocks')
    model_params_group.add_argument('--int_emb_size', type=int, default=64, help='Interaction embedding size')
    model_params_group.add_argument('--basis_emb_size', type=int, default=8, help='Basis embedding size')
    model_params_group.add_argument('--out_emb_channels', type=int, default=256, help='Output embedding channels')
    model_params_group.add_argument('--num_spherical', type=int, default=7, help='Number of spherical harmonics')
    model_params_group.add_argument('--num_radial', type=int, default=6, help='Number of radial basis functions')
    model_params_group.add_argument('--cutoff', type=float, default=5.0,
                                    help='Cutoff distance for neighbor calculation')
    model_params_group.add_argument('--envelope_exponent', type=int, default=5, help='Envelope exponent')
    model_params_group.add_argument('--num_before_skip', type=int, default=1,
                                    help='Number of layers before skip connection')
    model_params_group.add_argument('--num_after_skip', type=int, default=2,
                                    help='Number of layers after skip connection')
    model_params_group.add_argument('--num_output_layers', type=int, default=3, help='Number of output layers')
    model_params_group.add_argument('--max_num_neighbors', type=int, default=32, help='Maximum number of neighbors')

    readout_group = parser.add_argument_group('Readout Parameters')
    readout_group.add_argument('--set_transformer_hidden_dim', type=int, default=512,
                               help='Hidden dimension for set transformer')
    readout_group.add_argument('--set_transformer_num_heads', type=int, default=16, help='Number of attention heads')
    readout_group.add_argument('--set_transformer_num_sabs', type=int, default=2, help='Number of Set Attention Blocks')
    readout_group.add_argument('--attention_hidden_dim', type=int, default=256, help='Hidden dimension for attention')
    readout_group.add_argument('--attention_num_heads', type=int, default=8, help='Number of attention heads')

    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    train_group.add_argument('--eval_batch_size', type=int, default=None, help='Batch size for evaluation')
    train_group.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs')
    train_group.add_argument('--min_epochs', type=int, default=10, help='Minimum number of epochs')
    train_group.add_argument('--early_stopping_patience', type=int, default=20, help='Early stopping patience')
    train_group.add_argument('--early_stopping_min_delta', type=float, default=0.0001,
                             help='Minimum improvement for early stopping')
    train_group.add_argument('--loss_function', type=str, default='mse', choices=SUPPORTED_LOSS_FUNCTIONS,
                             help='Loss function')
    train_group.add_argument('--target_weights', type=float, nargs='+', default=None,
                             help='Weights for each target in loss calculation')
    train_group.add_argument('--uncertainty_method', type=str, default=None,
                             choices=[None, 'ensemble', 'dropout', 'evidential'])
    train_group.add_argument('--gradient_clip_val', type=float, default=0.0, help='Gradient clipping value')
    train_group.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')

    optim_group = parser.add_argument_group('Optimization Parameters')
    optim_group.add_argument('--optimizer', type=str, default='adam', choices=SUPPORTED_OPTIMIZERS, help='Optimizer')
    optim_group.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    optim_group.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    optim_group.add_argument('--scheduler', type=str, default='cosine', choices=SUPPORTED_LR_SCHEDULERS,
                             help='LR scheduler')
    optim_group.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs')
    optim_group.add_argument('--scheduler_patience', type=int, default=5, help='Scheduler patience')
    optim_group.add_argument('--scheduler_factor', type=float, default=0.5, help='Scheduler factor')
    optim_group.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')

    output_group = parser.add_argument_group('Output Parameters')
    output_group.add_argument('--out_dir', type=str, default='./results', help='Output directory')
    output_group.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    output_group.add_argument('--save_best_model', action='store_true', default=True, help='Save best model')
    output_group.add_argument('--save_last_model', action='store_true', default=False, help='Save last model')
    output_group.add_argument('--save_predictions', action='store_true', default=True, help='Save predictions')
    output_group.add_argument('--save_interval', type=int, default=0, help='Save checkpoints every N epochs')
    output_group.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint')

    dist_group = parser.add_argument_group('Distributed Training Parameters')
    dist_group.add_argument('--strategy', type=str, default='auto',
                            choices=['auto', 'ddp', 'deepspeed', 'fsdp', 'none'])
    dist_group.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    dist_group.add_argument('--devices', type=int, default=1, help='Number of devices per node')

    logging_group = parser.add_argument_group('Logging Parameters')
    logging_group.add_argument('--log_level', type=str, default='info',
                               choices=['debug', 'info', 'warning', 'error', 'critical'])
    logging_group.add_argument('--log_to_file', action='store_true', default=False, help='Log to file')
    logging_group.add_argument('--log_every_n_steps', type=int, default=50, help='Log every N steps')
    logging_group.add_argument('--logger_type', type=str, default='tensorboard',
                               choices=['tensorboard', 'wandb', 'csv', 'all'])
    logging_group.add_argument('--progress_bar', action='store_true', default=True, help='Show progress bar')

    dataset_specific_group = parser.add_argument_group('Dataset-Specific Parameters')
    dataset_specific_group.add_argument('--reaction_dataset_root', type=str, default=DEFAULT_REACTION_DATASET_ROOT)
    dataset_specific_group.add_argument('--reaction_target_fields', type=str, nargs='+', default=None,
                                        help='Target field(s) to predict')
    dataset_specific_group.add_argument('--reaction_file_suffixes', type=str, nargs=3,
                                        default=['_reactant.xyz', '_ts.xyz', '_product.xyz'])
    dataset_specific_group.add_argument('--input_features', type=str, nargs='+', default=['G(TS)_xtb', 'DrG_xtb'],
                                        help='Input feature columns to read from CSV')

    return parser


def process_args(parser=None):
    if parser is None:
        parser = get_parser()

    args = parser.parse_args()

    if args.config is not None:
        config_dict = load_config(args.config)
        for key, value in config_dict.items():
            if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

    validate_args(args)
    process_derived_args(args)

    return args


def validate_args(args):
    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Choose from {SUPPORTED_DATASETS}")

    if args.model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model type: {args.model_type}. Choose from {SUPPORTED_MODELS}")

    using_separate_files = args.val_csv is not None and args.test_csv is not None

    if not using_separate_files and args.cv_folds == 0:
        split_sum = args.train_ratio + args.val_ratio + args.test_ratio
        if abs(split_sum - 1.0) > 1e-6:
            raise ValueError(f"Train, validation, and test ratios must sum to 1.0, got {split_sum}")

        if not args.dataset_csv:
            raise ValueError("dataset_csv is required when using automatic splitting")

    else:
        if not args.dataset_csv:
            raise ValueError(
                "dataset_csv is required as the training dataset when using separate validation and test files")

        if not os.path.exists(args.dataset_csv):
            raise ValueError(f"Dataset CSV file does not exist: {args.dataset_csv}")

        if using_separate_files:
            for csv_file, name in [(args.val_csv, "validation"), (args.test_csv, "test")]:
                if not os.path.exists(csv_file):
                    raise ValueError(f"{name.capitalize()} CSV file does not exist: {csv_file}")

    if args.batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {args.batch_size}")

    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        raise ValueError(f"Evaluation batch size must be positive, got {args.eval_batch_size}")

    if args.lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {args.lr}")

    if args.max_epochs < args.min_epochs:
        raise ValueError(f"Maximum epochs ({args.max_epochs}) must be >= minimum epochs ({args.min_epochs})")

    if args.precision not in ['16', '32', 'bf16', 'mixed']:
        raise ValueError(f"Precision must be one of ['16', '32', 'bf16', 'mixed'], got {args.precision}")

    if args.cv_folds < 0:
        raise ValueError(f"Number of CV folds must be non-negative, got {args.cv_folds}")

    if args.cv_folds > 0 and args.cv_test_fold >= args.cv_folds:
        raise ValueError(f"Test fold index ({args.cv_test_fold}) must be less than number of folds ({args.cv_folds})")

    if args.dataset == 'XTB':
        if not os.path.exists(args.reaction_dataset_root):
            raise ValueError(f"Reaction dataset root directory does not exist: {args.reaction_dataset_root}")

        if not using_separate_files and not os.path.exists(args.dataset_csv):
            raise ValueError(f"Dataset CSV file does not exist: {args.dataset_csv}")

        if args.reaction_file_suffixes is not None and len(args.reaction_file_suffixes) != 3:
            raise ValueError(f"reaction_file_suffixes must specify exactly 3 suffixes")

        if args.target_weights is not None and args.reaction_target_fields is not None:
            if len(args.target_weights) != len(args.reaction_target_fields):
                raise ValueError(
                    f"Number of target weights ({len(args.target_weights)}) must match number of target fields ({len(args.reaction_target_fields)})")


def process_derived_args(args):
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.dataset}_{args.model_type}_{args.readout}_seed{args.random_seed}_{timestamp}"

    args.output_dir = os.path.join(
        args.out_dir,
        args.experiment_name
    )

    args.max_num_atoms = MAX_NUM_ATOMS_IN_MOL.get(args.dataset, 100)

    if args.reaction_target_fields is not None and args.target_weights is None:
        args.target_weights = [1.0] * len(args.reaction_target_fields)


def load_config(config_path):
    if not os.path.exists(config_path):
        raise ValueError(f"Configuration file does not exist: {config_path}")

    file_ext = os.path.splitext(config_path)[1].lower()

    if file_ext in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif file_ext == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {file_ext}. Use .yaml, .yml, or .json")


def save_config(args, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    config_dict = vars(args)

    yaml_path = os.path.join(output_dir, 'config.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    json_path = os.path.join(output_dir, 'config.json')
    with open(json_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    return yaml_path


def setup_logging(args):
    os.makedirs(args.output_dir, exist_ok=True)

    log_level = getattr(logging, args.log_level.upper())

    class DuplicateFilter(logging.Filter):
        def __init__(self):
            super().__init__()
            self.seen_messages = set()

        def filter(self, record):
            message = record.getMessage()
            if message in self.seen_messages:
                return False
            self.seen_messages.add(message)
            return True

    logger = logging.getLogger('deep')
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    duplicate_filter = DuplicateFilter()
    logger.addFilter(duplicate_filter)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if args.log_to_file:
        log_file = os.path.join(args.output_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_experiment_config(args):
    config = {
        'dataset': args.dataset,
        'model_type': args.model_type,
        'readout': args.readout,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'optimizer': args.optimizer,
        'learning_rate': args.lr,
        'random_seed': args.random_seed,
    }

    model_specific_config = {
        'hidden_channels': args.hidden_channels,
        'num_blocks': args.num_blocks,
        'int_emb_size': args.int_emb_size,
        'basis_emb_size': args.basis_emb_size,
        'out_emb_channels': args.out_emb_channels,
        'num_spherical': args.num_spherical,
        'num_radial': args.num_radial,
        'cutoff': args.cutoff,
        'envelope_exponent': args.envelope_exponent,
    }

    config.update(model_specific_config)

    if args.readout == 'set_transformer':
        config.update({
            'set_transformer_hidden_dim': args.set_transformer_hidden_dim,
            'set_transformer_num_heads': args.set_transformer_num_heads,
            'set_transformer_num_sabs': args.set_transformer_num_sabs,
        })

    if args.dataset == 'XTB':
        config.update({
            'reaction_dataset_root': args.reaction_dataset_root,
            'dataset_csv': args.dataset_csv,
        })

        if args.val_csv and args.test_csv:
            config.update({
                'val_csv': args.val_csv,
                'test_csv': args.test_csv,
            })
        else:
            config.update({
                'train_ratio': args.train_ratio,
                'val_ratio': args.val_ratio,
                'test_ratio': args.test_ratio,
            })

        config.update({
            'reaction_target_fields': args.reaction_target_fields,
            'reaction_file_suffixes': args.reaction_file_suffixes,
            'input_features': args.input_features
        })

    if args.cv_folds > 0:
        config.update({
            'cv_folds': args.cv_folds,
            'cv_test_fold': args.cv_test_fold,
            'cv_stratify': args.cv_stratify,
            'cv_grouped': args.cv_grouped
        })

    return config