#!/usr/bin/env python
import argparse
import os
import sys
import torch
from deepreaction import ReactionPredictor, ReactionDataset

def main():
    parser = argparse.ArgumentParser(description='Make predictions with a trained molecular reaction model')
    parser.add_argument('--checkpoint_path', type=str, 
                        default='./results/reaction_model/checkpoints/best-epoch=0000-val_total_loss=0.4343.ckpt',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--dataset_root', type=str, default='./dataset/DATASET_DA_F',
                        help='Dataset root directory')
    parser.add_argument('--dataset_csv', type=str, default='./dataset/DATASET_DA_F/dataset_xtb_final.csv',
                        help='Dataset CSV file')
    parser.add_argument('--input_features', type=str, nargs='+', default=['G(TS)_xtb', 'DrG_xtb'],
                        help='Input features')
    parser.add_argument('--file_patterns', type=str, nargs='+', default=['*_reactant.xyz', '*_ts.xyz', '*_product.xyz'],
                        help='File patterns')
    parser.add_argument('--id_field', type=str, default='ID', help='ID field name')
    parser.add_argument('--dir_field', type=str, default='R_dir', help='Directory field name')
    parser.add_argument('--reaction_field', type=str, default='reaction', help='Reaction field name')
    parser.add_argument('--output_csv', type=str, default='./predictions.csv',
                        help='Path to save predictions CSV')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Directory to save prediction output')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='Do not use CUDA')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    args = parser.parse_args()
    
    # Setup device (GPU/CPU)
    if args.cuda and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
        print("Using CPU")
        args.cuda = False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory set to: {args.output_dir}")
    
    # Create predictor
    predictor = ReactionPredictor(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gpu=args.cuda,
        num_workers=args.num_workers
    )
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_root} using CSV {args.dataset_csv}")
    dataset = ReactionDataset(
        root=args.dataset_root,
        csv_file=args.dataset_csv,
        target_fields=None,  # Not needed for inference
        file_patterns=args.file_patterns,
        input_features=args.input_features,
        id_field=args.id_field,
        dir_field=args.dir_field,
        reaction_field=args.reaction_field,
        inference_mode=True
    )
    
    print(f"Dataset loaded successfully")
    
    # Run inference
    print("Running inference...")
    results_df = predictor.predict_from_dataset(
        dataset=dataset,
        csv_output_path=args.output_csv
    )
    
    print(f"\nPredictions successfully saved to: {args.output_csv}")
    print(f"Total number of predictions generated: {len(results_df)}")
    
    if len(results_df) > 0:
        print("\nSample predictions (first 5 rows):")
        print(results_df.head())
    
    return 0

if __name__ == "__main__":
    sys.exit(main())