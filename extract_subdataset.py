import torch
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def extract_subdataset():
    # 读取原始数据
    user_counts = defaultdict(int)
    item_counts = defaultdict(int)
    interactions = []
    
    with open('d:/github/Opt/ratings.dat') as f:
        for line in f:
            user, item, rating, _ = line.strip().split('::')
            user = int(user)
            item = int(item)
            user_counts[user] += 1
            item_counts[item] += 1
            interactions.append((user, item, rating))
    
    # 筛选活跃用户和热门商品
    active_users = sorted(user_counts.keys(), key=lambda x: -user_counts[x])[:50]
    popular_items = sorted(item_counts.keys(), key=lambda x: -item_counts[x])[:50]
    
    # 获取其他用户和商品
    other_users = [u for u in user_counts.keys() if u not in active_users]
    other_items = [i for i in item_counts.keys() if i not in popular_items]
    
    # 构建训练集
    train_interactions = []
    for user, item, rating in interactions:
        if user in active_users and item in popular_items:
            train_interactions.append((user, item, rating))
    
    # 构建测试集1（50个活跃用户对其他商品的评分）
    test_user_interactions = []
    for user, item, rating in interactions:
        if user in active_users and item in other_items:
            test_user_interactions.append((user, item, rating))
    
    # 从测试集1中随机选择每个用户2个商品的评分
    import random
    test_user_final = []
    for user in active_users:
        user_ratings = [(u,i,r) for u,i,r in test_user_interactions if u == user]
        if len(user_ratings) > 2:
            test_user_final.extend(random.sample(user_ratings, 2))
        else:
            test_user_final.extend(user_ratings)
    
    # 构建测试集2（其他用户对50个热门商品的评分）
    test_item_interactions = []
    for user, item, rating in interactions:
        if user in other_users and item in popular_items:
            test_item_interactions.append((user, item, rating))
    
    # 随机选择2个其他用户
    selected_other_users = random.sample(other_users, 2)
    test_item_final = []
    for user in selected_other_users:
        user_ratings = [(u,i,r) for u,i,r in test_item_interactions if u == user]
        test_item_final.extend(user_ratings)
    
    # 重新映射ID
    user_map = {u: i+1 for i, u in enumerate(active_users)}
    item_map = {i: j+1 for j, i in enumerate(popular_items)}
    
    # 写入训练集数据文件
    with open('d:/github/Opt/train_ratings.dat', 'w') as f:
        for user, item, rating in train_interactions:
            f.write(f"{user_map[user]}::{item_map[item]}::{rating}\n")
    
    print(f"提取完成: 共{len(user_map)}个用户, {len(item_map)}个商品, {len(train_interactions)}条评分记录")

    # 构建训练集DataFrame
    train_df = pd.DataFrame(index=range(1, len(user_map)+1),
                     columns=range(1, len(item_map)+1))
    
    # 使用重新映射后的ID填充DataFrame
    for user, item, rating in train_interactions:
        mapped_user = user_map[user]
        mapped_item = item_map[item]
        train_df.at[mapped_user, mapped_item] = float(rating)
    
    # 可视化训练集矩阵
    plt.figure(figsize=(12, 8))
    plt.table(cellText=train_df.fillna('').values,
             rowLabels=train_df.index,
             colLabels=train_df.columns,
             cellLoc='center',
             loc='center')
    plt.axis('off')
    plt.tight_layout()
    
    # 保存训练集矩阵图片
    plt.savefig('d:/github/Opt/rating_matrix.png', dpi=300, bbox_inches='tight')
    print("评分矩阵可视化已保存为 rating_matrix.png")
    plt.close()

    # 构建测试集1 DataFrame并保存为CSV
    test_user_df = pd.DataFrame(index=range(1, len(user_map)+1),
                               columns=['item1', 'item2'])
    for user, item, rating in test_user_final:
        mapped_user = user_map[user]
        # 找到该用户已有的空列
        if pd.isna(test_user_df.at[mapped_user, 'item1']):
            test_user_df.at[mapped_user, 'item1'] = f"{item}({rating})"
        else:
            test_user_df.at[mapped_user, 'item2'] = f"{item}({rating})"
    test_user_df.to_csv('d:/github/Opt/test_user_ratings.csv')

    # 构建测试集2 DataFrame并保存为CSV
    test_item_df = pd.DataFrame(index=['user1', 'user2'],
                               columns=range(1, len(item_map)+1))
    current_user_row = 'user1'
    current_user = None
    for user, item, rating in test_item_final:
        if current_user != user:
            current_user = user
            current_user_row = 'user2' if current_user_row == 'user1' else 'user1'
        mapped_item = item_map[item]
        test_item_df.at[current_user_row, mapped_item] = f"{user}({rating})"
    test_item_df.to_csv('d:/github/Opt/test_item_ratings.csv')

    print(f"数据集提取完成:")
    print(f"训练集: {len(user_map)}个用户, {len(item_map)}个商品, {len(train_interactions)}条评分记录")
    print(f"测试集1(活跃用户对新商品): {len(test_user_final)}条评分记录")
    print(f"测试集2(新用户对热门商品): {len(test_item_final)}条评分记录")

    # 可视化训练集
    plt.figure(figsize=(12, 8))
    plt.table(cellText=train_df.fillna('').values,
             rowLabels=train_df.index,
             colLabels=train_df.columns,
             cellLoc='center',
             loc='center')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('d:/github/Opt/rating_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
if __name__ == '__main__':
    extract_subdataset()