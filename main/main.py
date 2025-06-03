import torch
import torch.optim as optim
import os
import random
import numpy as np
import pandas as pd 
import itertools 
import wandb 
from collections import defaultdict # 确保导入

from parse import parse_args
from datasets.dataset_dataloader import RatingMatrixDataset, get_iterative_rating_dataloader, get_test_dataloader_from_tuples 
from models.mf import MatrixFactorization 
from train import run_training_iteration 
from decoupled.tracin_cp import TracInCP_Decoupled, evaluate_model_on_test_set 
from baselines import get_random_collection_points, get_active_user_collection_points, get_popular_item_collection_points


import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['SimSun']
import seaborn as sns


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_collection_points_visualization(
    num_users, num_items, existing_ratings_set, collection_points_list, iteration_num, strategy_name, output_dir, 
    test_data_tuples=None  # 新增参数
    ):
    if num_users == 0 or num_items == 0:
        print(f"Iter {iteration_num} ({strategy_name}): 用户数或物品数为0，无法生成可视化图像。")
        return
    
    iter_output_dir = os.path.join(output_dir, f"strategy_{strategy_name}") 
    os.makedirs(iter_output_dir, exist_ok=True)
    output_path = os.path.join(iter_output_dir, f"rating_matrix_iter_{iteration_num}_collection.png")

    matrix_size = num_users * num_items
    heatmap_threshold = 500 * 500  

    if matrix_size > heatmap_threshold:
        # 简化版可视化的情况
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_title(f"Iter {iteration_num} ({strategy_name}): Collection Points (Size: {num_users}x{num_items})", fontsize=12)
        ax.set_xlabel("Item ID (0-based)", fontsize=10)
        ax.set_ylabel("User ID (0-based)", fontsize=10)
        
        # 添加测试集点的显示
        if test_data_tuples:
            test_users, test_items, _ = zip(*test_data_tuples)
            ax.scatter(test_items, test_users, color='blue', marker='o', s=30, alpha=0.5, 
                      label=f'Test Set ({len(test_data_tuples)})')

        if collection_points_list:
            users_cp, items_cp = zip(*collection_points_list)
            ax.scatter(items_cp, users_cp, color='red', marker='x', s=50, 
                      label=f'To Collect ({len(collection_points_list)})')
        
        # 显示现有评分
        sample_existing_limit = 20000
        if existing_ratings_set:
            valid_existing_ratings = [point for point in existing_ratings_set if isinstance(point, tuple) and len(point) == 2]
            if len(valid_existing_ratings) > sample_existing_limit:
                sampled_existing = random.sample(valid_existing_ratings, sample_existing_limit)
                if sampled_existing:
                    users_ex, items_ex = zip(*sampled_existing)
                    ax.scatter(items_ex, users_ex, color='green', marker='.', alpha=0.2, s=10, 
                             label=f'Existing (Sampled {sample_existing_limit})')
            elif valid_existing_ratings:
                users_ex, items_ex = zip(*valid_existing_ratings)
                ax.scatter(items_ex, users_ex, color='green', marker='.', alpha=0.3, s=10, 
                          label=f'Existing ({len(valid_existing_ratings)})')

        ax.set_xlim(-0.5, num_items - 0.5 if num_items > 0 else 0.5)
        ax.set_ylim(-0.5, num_users - 0.5 if num_users > 0 else 0.5)
        ax.invert_yaxis()
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

    else:
        # 完整热力图的情况
        rating_matrix_display = np.full((num_users, num_items), 0.5)  # 灰色背景

        # 标记现有评分
        for r_user, r_item in existing_ratings_set:
            if 0 <= r_user < num_users and 0 <= r_item < num_items:
                rating_matrix_display[r_user, r_item] = 1.0  # 绿色

        # 标记待采集点
        for cp_user, cp_item in collection_points_list:
            if 0 <= cp_user < num_users and 0 <= cp_item < num_items:
                rating_matrix_display[cp_user, cp_item] = 0.0  # 红色

        # 标记测试集点
        if test_data_tuples:
            for test_user, test_item, _ in test_data_tuples:
                if 0 <= test_user < num_users and 0 <= test_item < num_items:
                    rating_matrix_display[test_user, test_item] = 0.25  # 蓝色

        plt.figure(figsize=(max(12, num_items / 4), max(10, num_users / 4)))

        # 更新颜色映射以包含测试集的颜色
        cmap = matplotlib.colors.ListedColormap(['red', 'blue', 'lightgray', 'green'])
        bounds = [-0.25, 0.125, 0.375, 0.75, 1.25]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        sns.heatmap(rating_matrix_display, cmap=cmap, norm=norm, cbar=False, linewidths=.1)

        plt.title(f"Iter {iteration_num} ({strategy_name}): 用户-物品评分矩阵 ({num_users}x{num_items})\n" + 
                 f"红色=待采集, 蓝色=测试集, 绿色=已有, 灰色=空缺", fontsize=12)
        plt.xlabel("物品 ID (0-based)", fontsize=10)
        plt.ylabel("用户 ID (0-based)", fontsize=10)

        step_items = max(1, num_items // 20 if num_items > 20 else 1)
        step_users = max(1, num_users // 20 if num_users > 20 else 1)

        if num_items > 0:
            plt.xticks(np.arange(0.5, num_items + 0.5, step=step_items),
                      labels=np.arange(0, num_items, step=step_items),
                      rotation=45, ha="right", fontsize=8)
        if num_users > 0:
            plt.yticks(np.arange(0.5, num_users + 0.5, step=step_users),
                      labels=np.arange(0, num_users, step=step_users),
                      rotation=0, fontsize=8)

        plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=150)
        print(f"Iter {iteration_num} ({strategy_name}): 评分矩阵可视化已保存至: {output_path}")
    except Exception as e:
        print(f"Iter {iteration_num} ({strategy_name}): 保存可视化图像失败: {e}")
    plt.close() 

def standardize_influences(influence_dict):
    """对影响值进行Z-score标准化"""
    if not influence_dict:
        return defaultdict(float)
    
    values = np.array(list(influence_dict.values()))
    if len(values) < 2: # 如果只有一个值或没有值，无法计算标准差
        return defaultdict(float, {k: 0.0 for k in influence_dict.keys()})

    mean = np.mean(values)
    std = np.std(values)

    standardized_dict = defaultdict(float)
    if std == 0: # 如果标准差为0（所有值相同）
        for k, v in influence_dict.items():
            standardized_dict[k] = 0.0 # 或者 (v - mean) 如果希望保留相对大小（如果所有值非0）
    else:
        for k, v in influence_dict.items():
            standardized_dict[k] = (v - mean) / std
    return standardized_dict


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if not args.disable_wandb:
        try:
            run_name = f"{args.active_learning_strategy}_iters{args.num_iterations}_pts{args.num_collection_points}_seed{args.seed}"
            if args.test_data_path: 
                test_file_name = os.path.basename(args.test_data_path).split('.')[0]
                run_name += f"_test_{test_file_name}"

            wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=args)
            print("Weights & Biases 初始化成功。")
        except Exception as e:
            print(f"Weights & Biases 初始化失败: {e}。将禁用W&B。")
            args.disable_wandb = True


    print(f"--- 0. 加载原始完整数据从: {args.original_data_path} ---")
    if not os.path.exists(args.original_data_path):
        print(f"错误: 原始数据文件 '{args.original_data_path}' 未找到。")
        print("正在创建一个虚拟的 'sub_ratings.csv' 以便演示...")
        dummy_users_orig = 50
        dummy_items_orig = 30
        dummy_matrix_orig_values = np.random.choice(list(range(1,6)) + [np.nan], 
                                               size=(dummy_users_orig, dummy_items_orig), 
                                               p=[0.08]*5 + [0.6])
        dummy_df_for_sub = pd.DataFrame(dummy_matrix_orig_values, 
                                index=range(1, dummy_users_orig + 1), 
                                columns=range(1, dummy_items_orig + 1))
        dummy_df_for_sub.to_csv('sub_ratings.csv', index_label='')
        args.original_data_path = 'sub_ratings.csv'

    current_dataset = RatingMatrixDataset(
        args.original_data_path,
        test_data_path=args.test_data_path 
    )
    
    args.num_users = current_dataset.num_users_original 
    args.num_items = current_dataset.num_items_original 

    if args.num_users == 0 or args.num_items == 0:
        print("错误：从原始数据推断的用户数或物品数为0。请检查原始数据文件。")
        if not args.disable_wandb: wandb.finish()
        return
    print(f"模型将基于原始数据维度初始化: 用户数={args.num_users}, 物品数={args.num_items}")

    test_loader = None
    test_data_tuples_from_split = current_dataset.get_test_data_tuples() 
    if test_data_tuples_from_split:
        test_loader = get_test_dataloader_from_tuples(
            test_data_tuples_from_split, 
            args.num_users, 
            args.num_items, 
            args.test_batch_size
        )
        if test_loader:
             print(f"测试集创建成功，包含 {len(test_loader.dataset)} 个样本。")
        else:
            print("警告: 测试集为空或创建失败 (从 get_test_data_tuples 返回的元组)。")
    else:
        print(f"警告: 未从 RatingMatrixDataset 中获取到测试数据元组 (test_data_path: {args.test_data_path})。")


    newly_collected_for_next_iter = [] 
    global_epoch_count = 0 

    for iter_num in range(1, args.num_iterations + 1):
        print(f"\n======= 主动学习迭代轮次: {iter_num}/{args.num_iterations} (策略: {args.active_learning_strategy}) =======")
        
        current_iter_wandb_metrics = {"iteration": iter_num, "strategy": args.active_learning_strategy}


        print(f"--- Iter {iter_num}: 准备数据 ---")
        if iter_num == 1:
            print("  生成初始稀疏训练数据...")
            current_dataset.generate_initial_sparse_matrix(args.user_retention_config)
        else:
            if newly_collected_for_next_iter:
                print(f"  将上一轮收集的 {len(newly_collected_for_next_iter)} 个评分添加到训练数据...")
                current_dataset.add_new_ratings(newly_collected_for_next_iter)
                current_iter_wandb_metrics["num_newly_added_ratings_this_iter"] = len(newly_collected_for_next_iter)
                newly_collected_for_next_iter = [] 
            else:
                print("  上一轮没有新的评分可以添加。")
                current_iter_wandb_metrics["num_newly_added_ratings_this_iter"] = 0
        
        current_iter_wandb_metrics["current_total_training_samples"] = len(current_dataset.data_tuples)

        if len(current_dataset.data_tuples) == 0 and iter_num > 1: 
            print(f"Iter {iter_num}: 当前训练数据为空，无法继续。")
            if not args.disable_wandb: wandb.log(current_iter_wandb_metrics, step=global_epoch_count) 
            break 
        
        train_loader = get_iterative_rating_dataloader(
            dataset_instance=current_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        print(f"--- Iter {iter_num}: 初始化/重置模型 ---")
        if args.model_name == 'MF':
            model = MatrixFactorization(args.num_users, args.num_items, args.embedding_dim).to(device)
            model_constructor_args = {
                'num_users': args.num_users, 
                'num_items': args.num_items, 
                'embedding_dim': args.embedding_dim
            }
            model_class_for_tracin = MatrixFactorization
        else:
            raise ValueError(f"不支持的模型名称: {args.model_name}")

        criterion = torch.nn.MSELoss() 
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) 

        print(f"--- Iter {iter_num}: 训练模型 ---")
        gradient_log_paths_map_iter = run_training_iteration(
            model, train_loader, test_loader, criterion, optimizer, args, iter_num, global_epoch_count, device
        )
        global_epoch_count += args.epochs_per_iteration 


        collection_points_this_iter = []
        if args.active_learning_strategy == 'TracInCP':
            print(f"--- Iter {iter_num}: 计算 TracInCP 解耦影响值 ---")
            unit_user_influences, unit_item_influences = defaultdict(float), defaultdict(float)


            if not gradient_log_paths_map_iter:
                print(f"Iter {iter_num}: TracInCP策略：没有checkpoints被保存，无法计算影响。")
            elif not current_dataset.data_tuples: 
                print(f"Iter {iter_num}: TracInCP策略：当前训练数据为空，无法进行影响计算。")
            elif not test_loader or len(test_loader.dataset) == 0:
                print(f"Iter {iter_num}: TracInCP策略：测试集为空，无法计算对测试集误差的影响。")
            else:
                current_checkpoint_dir_iter = os.path.join(args.checkpoint_dir_base, f"strategy_{args.active_learning_strategy}", f"iter_{iter_num}")
                tracin_analyzer = TracInCP_Decoupled(
                    model_class_constructor=model_class_for_tracin, 
                    model_args=model_constructor_args,             
                    task_type=args.task_type,
                    current_checkpoint_dir=current_checkpoint_dir_iter, 
                    current_gradient_log_paths_map=gradient_log_paths_map_iter, 
                    criterion=criterion
                )
                
                print(f"  TracInCP: 使用整个测试集 (包含 {len(test_loader.dataset)} 个样本) 作为影响目标。")
                # compute_cumulative_influences_on_test_set 返回的是单位影响值
                unit_user_influences, unit_item_influences = tracin_analyzer.compute_cumulative_influences_on_test_set(
                    test_loader 
                )
            
            if unit_user_influences or unit_item_influences: 
                # 标准化影响值
                norm_user_influences = standardize_influences(unit_user_influences)
                norm_item_influences = standardize_influences(unit_item_influences)
                print(f"  TracInCP: 用户影响力已标准化 (均值~0, 标准差~1 如果数据多样)。")
                print(f"  TracInCP: 物品影响力已标准化。")


                candidate_pairs_with_score = []
                
                users_to_consider_for_ranking = list(norm_user_influences.keys()) if norm_user_influences else list(range(args.num_users))
                items_to_consider_for_ranking = list(norm_item_influences.keys()) if norm_item_influences else list(range(args.num_items))
                
                # print(f"  TracInCP: 考虑 {len(users_to_consider_for_ranking)} 个有影响的用户和 {len(items_to_consider_for_ranking)} 个有影响的物品进行组合...")
                
                MAX_USERS_FOR_PRODUCT = 2000 
                MAX_ITEMS_FOR_PRODUCT = 2000
                
                if len(users_to_consider_for_ranking) > MAX_USERS_FOR_PRODUCT:
                    sorted_users_by_abs_inf = sorted(users_to_consider_for_ranking, key=lambda u: abs(norm_user_influences.get(u,0.0)), reverse=True)
                    users_to_consider_for_ranking = sorted_users_by_abs_inf[:MAX_USERS_FOR_PRODUCT]
                if len(items_to_consider_for_ranking) > MAX_ITEMS_FOR_PRODUCT:
                    sorted_items_by_abs_inf = sorted(items_to_consider_for_ranking, key=lambda i: abs(norm_item_influences.get(i,0.0)), reverse=True)
                    items_to_consider_for_ranking = sorted_items_by_abs_inf[:MAX_ITEMS_FOR_PRODUCT]


                considered_pairs_count = 0
                max_considered_pairs = 1_000_000 
                
                for u_id in users_to_consider_for_ranking:
                    if considered_pairs_count > max_considered_pairs and len(candidate_pairs_with_score) > args.num_collection_points * 10: # 增加候选池大小
                        break
                    for i_id in items_to_consider_for_ranking:
                        considered_pairs_count +=1
                        if considered_pairs_count > max_considered_pairs and len(candidate_pairs_with_score) > args.num_collection_points * 10:
                            break
                        if (u_id, i_id) not in current_dataset.existing_ratings_set:
                            user_inf = norm_user_influences.get(u_id, 0.0) 
                            item_inf = norm_item_influences.get(i_id, 0.0) 
                            combined_score = 0.5 * user_inf + 0.5 * item_inf 
                            candidate_pairs_with_score.append(((u_id, i_id), combined_score))
                
                candidate_pairs_with_score.sort(key=lambda x: x[1], reverse=True) 

                for (user_id, item_id), score in candidate_pairs_with_score:
                    if (user_id, item_id) not in current_dataset.existing_ratings_set: 
                        collection_points_this_iter.append((user_id, item_id))
                        if len(collection_points_this_iter) >= args.num_collection_points:
                            break
                print(f"  Iter {iter_num} (TracInCP): 从 {len(candidate_pairs_with_score)} 个候选对中选择了采集点。")
            else: 
                print(f"  Iter {iter_num} (TracInCP): 影响分数计算失败或为空，回退到随机选择。")
                collection_points_this_iter = get_random_collection_points(
                    args.num_collection_points, args.num_users, args.num_items, current_dataset.existing_ratings_set
                )

        elif args.active_learning_strategy == 'Random':
            print(f"--- Iter {iter_num}: 使用随机策略生成采集点 ---")
            collection_points_this_iter = get_random_collection_points(
                args.num_collection_points, args.num_users, args.num_items, current_dataset.existing_ratings_set
            )
        elif args.active_learning_strategy == 'ActiveUser':
            print(f"--- Iter {iter_num}: 使用活跃用户策略生成采集点 ---")
            collection_points_this_iter = get_active_user_collection_points(
                args.num_collection_points, current_dataset.data_tuples, 
                args.num_users, args.num_items, current_dataset.existing_ratings_set
            )
        elif args.active_learning_strategy == 'PopularItem':
            print(f"--- Iter {iter_num}: 使用热门物品策略生成采集点 ---")
            collection_points_this_iter = get_popular_item_collection_points(
                args.num_collection_points, current_dataset.data_tuples,
                args.num_users, args.num_items, current_dataset.existing_ratings_set
            )
        
        print(f"  Iter {iter_num}: 已识别出 {len(collection_points_this_iter)} 个新的待采集位置 (目标: {args.num_collection_points}):")
        for i, (u, iid) in enumerate(collection_points_this_iter):
            if i < 5:
                if args.active_learning_strategy == 'TracInCP' and norm_user_influences and norm_item_influences:
                    user_inf = norm_user_influences.get(u, 0.0)
                    item_inf = norm_item_influences.get(iid, 0.0)
                    combined_inf = 0.5 * user_inf + 0.5 * item_inf
                    print(f"    用户 {u} (影响力: {user_inf:.4f}), 物品 {iid} (影响力: {item_inf:.4f}), 组合影响力: {combined_inf:.4f}")
                else:
                    print(f"    用户 {u}, 物品 {iid}")
            elif i == 5:
                print("    ...")
                break
        current_iter_wandb_metrics["num_candidate_collection_points"] = len(collection_points_this_iter)


        print(f"--- Iter {iter_num}: 模拟数据采集 ---")
        num_actually_collected = 0
        if collection_points_this_iter:
            for cp_user, cp_item in collection_points_this_iter:
                true_rating = current_dataset.get_original_rating_from_train_split(cp_user, cp_item)
                if true_rating != 0.0: 
                    newly_collected_for_next_iter.append((cp_user, cp_item, true_rating))
                    num_actually_collected +=1
            print(f"  Iter {iter_num}: 已为 {num_actually_collected} 个采集点获取了“真实”评分并准备添加到下一轮。")
        else:
            print(f"  Iter {iter_num}: 没有新的采集点可供收集评分。")
        current_iter_wandb_metrics["num_actually_collected_ratings_this_iter"] = num_actually_collected
        
        if not args.disable_wandb:
            final_test_loss_this_iter = 0.0
            if model and test_loader and len(test_loader.dataset) > 0 : 
                 final_test_loss_this_iter, _ = evaluate_model_on_test_set(model, test_loader, criterion, device, log_predictions_to_wandb=False) 
                 current_iter_wandb_metrics[f"FinalTestLoss_iteration_end"] = final_test_loss_this_iter
                 print(f"  Iter {iter_num}: 迭代结束时最终测试损失: {final_test_loss_this_iter:.6f}")
            else:
                 current_iter_wandb_metrics[f"FinalTestLoss_iteration_end"] = -1.0 
            
            wandb.log(current_iter_wandb_metrics, step=global_epoch_count) 


        print(f"--- Iter {iter_num}: 生成可视化图像 ---")
        generate_collection_points_visualization(
            args.num_users, 
            args.num_items, 
            current_dataset.existing_ratings_set, 
            collection_points_this_iter, 
            iter_num,
            args.active_learning_strategy, 
            args.output_visualization_dir,
            test_data_tuples=test_data_tuples_from_split  # 新增参数
        )
        
        if not newly_collected_for_next_iter and iter_num < args.num_iterations: 
            print(f"Iter {iter_num}: 没有新的有效评分被采集到，主动学习过程可能已收敛或无新信息可获取。")
            if not collection_points_this_iter : 
                print(f"Iter {iter_num}: 且没有新的候选采集点，提前结束迭代。")
                break 


    print(f"\n======= 主动学习所有 {args.num_iterations} 轮迭代完成 =======")
    if not args.disable_wandb:
        wandb.finish()
        print("W&B run finished.")
    print("--- 项目执行完毕 ---")

if __name__ == '__main__':
    main()
