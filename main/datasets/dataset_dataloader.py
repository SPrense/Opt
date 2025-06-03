import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import os

class RatingMatrixDataset(Dataset):
    def __init__(self, original_csv_path, test_data_path): # 移除了test_split_strategy等参数
        self.original_csv_path = original_csv_path
        self.full_original_df = self._load_original_df(original_csv_path) 
        
        self.num_users_original = 0
        self.num_items_original = 0
        if not self.full_original_df.empty:
            self.num_users_original = self.full_original_df.index.max() + 1
            self.num_items_original = self.full_original_df.columns.max() + 1
        
        print(f"完整原始数据集加载: {self.num_users_original} 用户 (max_id+1), {self.num_items_original} 物品 (max_id+1)。")

        self.original_train_df = self.full_original_df.copy() # 初始化为完整数据
        self.test_data_tuples = []          

        if test_data_path and os.path.exists(test_data_path):
            self._extract_test_data_and_update_train_df(test_data_path)
        else:
            if test_data_path:
                 print(f"警告: 指定的测试数据文件 '{test_data_path}' 未找到。将不使用特定测试集。")
            else:
                 print("未提供特定测试数据文件，所有数据（经稀疏化后）将用于训练。")


        self.data_tuples = [] 
        self.existing_ratings_set = set() 
        self.num_users_current = 0 
        self.num_items_current = 0 

    def _load_original_df(self, csv_path):
        try:
            df = pd.read_csv(csv_path, header=0, index_col=0)
            df.index = df.index.astype(int) - 1 
            df.columns = df.columns.astype(int) - 1
            if (df.index < 0).any() or (df.columns < 0).any():
                raise ValueError("CSV文件中的用户ID或物品ID在减1后变为负数。请确保ID从1开始。")
            return df
        except FileNotFoundError:
            print(f"错误：原始数据文件 {csv_path} 未找到。")
            raise
        except ValueError as ve:
            print(f"处理CSV文件 {csv_path} 时发生值错误: {ve}")
            raise
        except Exception as e:
            print(f"读取原始CSV文件 {csv_path} 时发生错误: {e}")
            raise
            
    def _extract_test_data_and_update_train_df(self, test_file_path):
        print(f"从 '{test_file_path}' 提取测试数据并更新训练集...")
        try:
            test_df_raw = pd.read_csv(test_file_path, header=0, index_col=0) # 假设第一列是索引
        except Exception as e:
            print(f"读取测试文件 {test_file_path} 失败: {e}。不使用此测试集。")
            return

        filename = os.path.basename(test_file_path).lower()
        
        if "test_col" in filename: # 例如 test_col.csv
            # 假设文件格式: index是物品ID (1-based), 列是用户ID (1-based), 值是评分
            # 但根据用户描述，test_col.csv: 第一列是item ID (50), then user IDs (1-50) as columns
            # 这意味着 test_df_raw 的 index 是 item_id (1-based), columns 是 user_id (1-based)
            # 我们需要转置它，或者直接按行读取
            # 假设 test_df_raw 只有一行，代表一个物品的评分
            if test_df_raw.shape[0] != 1:
                print(f"警告: {test_file_path} 格式不符合预期 (应只有一行代表一个物品)。跳过。")
                return
            
            test_item_id_1based = test_df_raw.index[0]
            test_item_id_0based = int(test_item_id_1based) - 1

            if not (0 <= test_item_id_0based < self.num_items_original):
                print(f"警告: 从 {test_file_path} 解析的物品ID {test_item_id_0based} 超出范围。跳过。")
                return

            print(f"  测试集针对物品 ID: {test_item_id_0based} (来自 {test_file_path})")
            
            # 遍历原始训练DF中该物品的所有评分
            if test_item_id_0based in self.original_train_df.columns:
                for user_id_0based in self.original_train_df.index:
                    rating = self.original_train_df.loc[user_id_0based, test_item_id_0based]
                    if pd.notna(rating):
                        self.test_data_tuples.append((user_id_0based, test_item_id_0based, float(rating)))
                        self.original_train_df.loc[user_id_0based, test_item_id_0based] = np.nan # 从训练集中移除
            else:
                print(f"  警告: 测试物品ID {test_item_id_0based} 不在原始训练DF的列中。")


        elif "test_row" in filename: # 例如 test_row.csv
            # 假设文件格式: index是用户ID (1-based), 列是物品ID (1-based), 值是评分
            # test_row.csv: 第一列是user ID (50), then item IDs (1-50) as columns
            if test_df_raw.shape[0] != 1:
                print(f"警告: {test_file_path} 格式不符合预期 (应只有一行代表一个用户)。跳过。")
                return

            test_user_id_1based = test_df_raw.index[0]
            test_user_id_0based = int(test_user_id_1based) - 1

            if not (0 <= test_user_id_0based < self.num_users_original):
                print(f"警告: 从 {test_file_path} 解析的用户ID {test_user_id_0based} 超出范围。跳过。")
                return
            
            print(f"  测试集针对用户 ID: {test_user_id_0based} (来自 {test_file_path})")

            # 遍历原始训练DF中该用户的所有评分
            if test_user_id_0based in self.original_train_df.index:
                # 获取所有可用的列并随机选择10个
                available_items = list(self.original_train_df.columns)
                num_items_to_sample = min(10, len(available_items))  # 确保不会超过可用的列数
                sampled_items = random.sample(available_items, num_items_to_sample)
                
                for item_id_0based in sampled_items:  # 只遍历随机选择的10个列
                    rating = self.original_train_df.loc[test_user_id_0based, item_id_0based]
                    if pd.notna(rating):
                        self.test_data_tuples.append((test_user_id_0based, item_id_0based, float(rating)))
                        self.original_train_df.loc[test_user_id_0based, item_id_0based] = np.nan  # 从训练集中移除
            else:
                print(f"  警告: 测试用户ID {test_user_id_0based} 不在原始训练DF的索引中。")
        else:
            print(f"警告: 未知的测试文件格式 '{test_file_path}'。跳过测试集提取。")
            return
            
        print(f"  从 '{test_file_path}' 提取了 {len(self.test_data_tuples)} 个测试样本。")
        print(f"  更新后的 original_train_df 用于生成稀疏训练数据。")


    def get_original_rating_from_train_split(self, user_id, item_id):
        if not self.original_train_df.empty and \
           user_id in self.original_train_df.index and \
           item_id in self.original_train_df.columns:
            rating = self.original_train_df.loc[user_id, item_id]
            return float(rating) if pd.notna(rating) else 0.0 
        return 0.0

    def generate_initial_sparse_matrix(self, user_retention_config):
        self.data_tuples = []
        self.existing_ratings_set = set()
        
        if self.original_train_df.empty: 
            print("警告：分割后的原始训练数据为空，无法生成稀疏矩阵。")
            self._update_current_dims()
            return

        all_user_indices = list(self.original_train_df.index)
        random.shuffle(all_user_indices) 

        start_user_idx_slice = 0
        temp_data_tuples = []

        for i in range(0, len(user_retention_config), 3):
            p_users = user_retention_config[i]
            min_ret = user_retention_config[i+1]
            max_ret = user_retention_config[i+2]

            num_users_in_train_df = len(all_user_indices)
            if num_users_in_train_df == 0: continue

            num_users_in_group = int(np.floor(p_users * num_users_in_train_df))
            if i + 3 >= len(user_retention_config): 
                num_users_in_group = num_users_in_train_df - start_user_idx_slice
            
            user_group = all_user_indices[start_user_idx_slice : start_user_idx_slice + num_users_in_group]
            start_user_idx_slice += num_users_in_group

            for user_id in user_group:
                if user_id not in self.original_train_df.index: continue
                user_ratings_series = self.original_train_df.loc[user_id].dropna()
                num_actual_ratings = len(user_ratings_series)
                if num_actual_ratings == 0:
                    continue

                retention_rate = random.uniform(min_ret, max_ret)
                num_to_keep = int(np.ceil(num_actual_ratings * retention_rate)) 
                num_to_keep = min(num_to_keep, num_actual_ratings) 
                
                if num_to_keep > 0 :
                    ratings_indices_to_keep = np.random.choice(user_ratings_series.index, size=num_to_keep, replace=False)
                    ratings_to_keep = user_ratings_series.loc[ratings_indices_to_keep]
                    for item_id, rating in ratings_to_keep.items(): 
                        temp_data_tuples.append((user_id, item_id, float(rating)))
                        self.existing_ratings_set.add((user_id, item_id))
        
        if not temp_data_tuples:
            print("警告：初始稀疏矩阵生成后为空。请检查retention config和原始数据。")
        
        self.data_tuples = temp_data_tuples
        self._update_current_dims() 
        print(f"初始稀疏训练数据已生成: {len(self.data_tuples)} 个评分。")
        print(f"  当前训练数据用户数 (基于0的最大ID+1): {self.num_users_current}, 物品数 (基于0的最大ID+1): {self.num_items_current}")


    def add_new_ratings(self, new_collected_ratings_list):
        added_count = 0
        for user_id, item_id, rating in new_collected_ratings_list:
            if (user_id, item_id) not in self.existing_ratings_set:
                if user_id < self.num_users_original and item_id < self.num_items_original:
                    self.data_tuples.append((user_id, item_id, float(rating)))
                    self.existing_ratings_set.add((user_id, item_id))
                    added_count +=1
                else:
                    print(f"警告: 尝试添加超出原始范围的评分 ({user_id}, {item_id})。跳过。")

        if added_count > 0:
            self._update_current_dims()
        print(f"已添加 {added_count} 个新评分到训练数据。总评分数: {len(self.data_tuples)}")
        print(f"  更新后训练数据用户数 (基于0的最大ID+1): {self.num_users_current}, 物品数 (基于0的最大ID+1): {self.num_items_current}")

    def _update_current_dims(self):
        if not self.data_tuples:
            self.num_users_current = 0 
            self.num_items_current = 0
            return

        all_user_ids_in_current_data = {uid for uid, _, _ in self.data_tuples}
        all_item_ids_in_current_data = {iid for _, iid, _ in self.data_tuples}
        
        self.num_users_current = max(all_user_ids_in_current_data) + 1 if all_user_ids_in_current_data else 0
        self.num_items_current = max(all_item_ids_in_current_data) + 1 if all_item_ids_in_current_data else 0

    def get_test_data_tuples(self):
        return self.test_data_tuples

    def __len__(self):
        return len(self.data_tuples) 

    def __getitem__(self, idx):
        user_id, item_id, rating = self.data_tuples[idx]
        return torch.LongTensor([user_id]), torch.LongTensor([item_id]), torch.FloatTensor([rating])

class TestDataset(Dataset):
    def __init__(self, test_data_tuples, num_total_users, num_total_items):
        self.data_tuples = []
        for u, i, r in test_data_tuples:
            if u < num_total_users and i < num_total_items:
                self.data_tuples.append((u,i,r))
        print(f"测试数据集初始化: {len(self.data_tuples)} 个有效评分。")


    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):
        user_id, item_id, rating = self.data_tuples[idx]
        return torch.LongTensor([user_id]), torch.LongTensor([item_id]), torch.FloatTensor([rating])


def get_iterative_rating_dataloader(dataset_instance, batch_size, shuffle=True):
    if len(dataset_instance) == 0: 
        empty_torch_dataset = torch.utils.data.TensorDataset(torch.empty(0,1, dtype=torch.long), torch.empty(0,1,dtype=torch.long), torch.empty(0,1,dtype=torch.float))
        return DataLoader(empty_torch_dataset, batch_size=batch_size)
    dataloader = DataLoader(dataset_instance, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def get_test_dataloader_from_tuples(test_data_tuples, num_total_users, num_total_items, batch_size):
    if not test_data_tuples:
        print(f"警告：提供的测试数据元组列表为空。")
        return None
    test_dataset = TestDataset(test_data_tuples, num_total_users, num_total_items)
    if len(test_dataset) == 0:
        print(f"警告：过滤后测试数据集为空。")
        return None
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)