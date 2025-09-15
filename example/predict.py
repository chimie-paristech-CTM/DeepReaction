import torch
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from deepreaction import Config, ReactionPredictor


def main():
    params = {
        'dataset': 'XTB',
        'readout': 'mean',
        'dataset_root': './dataset/DATASET_DA_F',
        'input_features': ['DG_act_xtb', 'DrG_xtb'],
        'file_patterns': ['*_reactant.xyz', '*_ts.xyz', '*_product.xyz'],
        'file_dir_pattern': 'reaction_*',
        'file_keywords': ['reactant', 'ts', 'product'],
        'id_field': 'ID',
        'dir_field': 'R_dir',
        'reaction_field': 'smiles',
        'use_scaler': True,
        'use_xtb_features': True,

        'batch_size': 32,
        'random_seed': 42234,

        'cuda': True,
        'gpu_id': 0,
        'num_workers': 4,
        'log_level': 'info'
    }

    checkpoint_path = "./results/reaction_model/checkpoints/best-epoch=0056-val_total_loss=0.0505.ckpt"

    inference_csv = "./dataset/DATASET_DA_F/dataset_xtb_final.csv"

    output_dir = "./predictions"

    if params['cuda'] and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['gpu_id'])
        device = torch.device(f"cuda:{params['gpu_id']}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
        print("Using CPU")
        params['cuda'] = False

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    if not os.path.exists(inference_csv):
        raise FileNotFoundError(f"Inference CSV not found: {inference_csv}")

    print(f"Loading model from: {checkpoint_path}")
    print(f"Inference data from: {inference_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Use XTB features: {params.get('use_xtb_features', True)}")

    config = Config.from_params(params)
    predictor = ReactionPredictor(config=config, checkpoint_path=checkpoint_path)

    print("Starting prediction...")
    results = predictor.predict_from_csv(inference_csv, output_dir=output_dir)

    print("Prediction completed successfully!")
    print(f"Results shape: {results.shape}")
    print(f"Results saved to: {output_dir}/predictions.csv")

    print("\nFirst 5 prediction results:")
    print(results.head())


if __name__ == "__main__":
    main()
