import argparse
import numpy as np 

def parse_args():
    parser = argparse.ArgumentParser(description="Iterative TracInCP Decoupled Influence Analysis")

    # --- 数据集参数 ---
    parser.add_argument('--original_data_path', type=str, default='sub_ratings.csv', 
                        help='Path to the original full dataset file (e.g., sub_ratings.csv)')
    
    # --- 测试集参数 ---
    parser.add_argument('--test_data_path', type=str, default='test_row.csv', 
                        help='Path to the test dataset file (e.g., test_col.csv or test_row.csv).')

    parser.add_argument('--num_users', type=int, default=50, 
                        help='Total number of users in the original dataset. Inferred if -1.')
    parser.add_argument('--num_items', type=int, default=50, 
                        help='Total number of items in the original dataset. Inferred if -1.')
    
    parser.add_argument('--user_retention_config', nargs='+', default=[
        0.05, 0.4, 0.8,
        0.15, 0.2, 0.4,
        0.30, 0.1, 0.2, 
        0.50, 0.0, 0.1 
    ], type=float, help='User rating retention config for initial sparse training data.')


    # --- 模型参数 ---
    parser.add_argument('--model_name', type=str, default='MF',
                        help='Name of the recommendation model to use (e.g., MF)')
    parser.add_argument('--embedding_dim', type=int, default=20,
                        help='Dimension of user and item embeddings')

    # --- 训练参数 ---
    parser.add_argument('--epochs_per_iteration', type=int, default=15, 
                        help='Number of training epochs per active learning iteration')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=10, 
                        help='Batch size for testing')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Learning rate for optimizer')
    parser.add_argument('--checkpoint_interval', type=int, default=5,
                        help='Interval (in epochs) for saving model checkpoints and gradients within an iteration')
    parser.add_argument('--checkpoint_dir_base', type=str, default='checkpoints_iter', 
                        help='Base directory to save model checkpoints for each iteration')
    parser.add_argument('--gradient_log_dir_base', type=str, default='gradient_logs_iter', 
                        help='Base directory to save gradient logs for TracInCP for each iteration')

    # --- TracInCP 与 主动学习策略参数 ---
    parser.add_argument('--active_learning_strategy', type=str, default='TracInCP',
                        choices=['TracInCP', 'Random', 'ActiveUser', 'PopularItem'],
                        help='Strategy for selecting new collection points.')
    parser.add_argument('--task_type', type=str, default='regression',
                        help='Task type for TracInCP (e.g., regression, classification)')
    parser.add_argument('--num_collection_points', type=int, default=50,
                        help='Number of new user-item pairs to identify for collection per iteration')
    parser.add_argument('--output_visualization_dir', type=str, default='visualizations_iter', 
                        help='Directory to save the visualization PNG images for each iteration')

    # --- 迭代学习参数 ---
    parser.add_argument('--num_iterations', type=int, default=8, 
                        help='Number of active learning iterations')

    # --- W&B 参数 ---
    parser.add_argument('--wandb_project', type=str, default="tracin_active_learning_v5", # 更新项目名
                        help="Weights & Biases project name.")
    parser.add_argument('--wandb_entity', type=str, default=None, 
                        help="Weights & Biases entity (username or team).")
    parser.add_argument('--disable_wandb', action='store_true',
                        help="Disable Weights & Biases logging.")


    # --- 其他 ---
    parser.add_argument('--seed', type=int, default=24, help='Random seed for reproducibility')

    args = parser.parse_args()
    return args