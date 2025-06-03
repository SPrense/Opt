import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

def load_sub_ratings():
    """从sub_ratings.dat加载数据"""
    user_counts = defaultdict(int)
    item_counts = defaultdict(int)
    interactions = []
    
    with open('d:/github/Opt/sub_ratings.dat') as f:
        for line in f:
            user, item, rating = line.strip().split('::')
            user = int(user)
            item = int(item)
            user_counts[user] += 1
            item_counts[item] += 1
            interactions.append((user, item, rating))
    
    return user_counts, item_counts, interactions

def create_normal_distribution_matrix():
    """创建符合特定比例分布的人工缺失矩阵"""
    user_counts, item_counts, interactions = load_sub_ratings()
    
    # 获取用户和商品列表
    users = sorted(user_counts.keys())
    items = sorted(item_counts.keys())
    
    # 创建完整评分矩阵
    df = pd.DataFrame(index=users, columns=items)
    for user, item, rating in interactions:
        df.at[user, item] = int(rating)
    
    # 随机打乱用户顺序
    np.random.shuffle(users)
    
    # 设置用户比例和对应的交互比例
    user_groups = [
        (0.05, 0.4),   # 5%用户保留>40%评分
        (0.15, 0.3),    # 15%用户保留20-40%评分
        (0.3, 0.1),     # 30%用户保留10%评分
        (0.5, 0.05)     # 50%用户保留<10%评分
    ]
    
    start = 0
    for ratio, keep_ratio in user_groups:
        end = start + int(len(users) * ratio)
        group_users = users[start:end]
        
        for user in group_users:
            user_items = df.loc[user].dropna().index.tolist()
            num_to_keep = max(1, int(len(user_items) * keep_ratio))
            
            if len(user_items) > num_to_keep:
                to_remove = np.random.choice(
                    user_items,
                    size=len(user_items)-num_to_keep,
                    replace=False
                )
                df.loc[user, to_remove] = np.nan
                
        start = end
    
    # 确保没有全空行或全空列
    while True:
        empty_rows = df.isnull().all(axis=1)
        empty_cols = df.isnull().all(axis=0)
        if not empty_rows.any() and not empty_cols.any():
            break
        # 对有问题的行/列随机恢复一个值
        for row in df.index[empty_rows]:
            col = np.random.choice(df.columns)
            df.at[row, col] = np.random.choice([1,2,3,4,5])
        for col in df.columns[empty_cols]:
            row = np.random.choice(df.index)
            df.at[row, col] = np.random.choice([1,2,3,4,5])
    
    return df

def visualize_matrix(matrix, title):
    """以表格形式可视化评分矩阵并保存为PNG"""
    plt.figure(figsize=(12, 8))
    plt.table(cellText=matrix.fillna('').values,
             rowLabels=matrix.index,
             colLabels=matrix.columns,
             cellLoc='center',
             loc='center')
    plt.axis('off')
    plt.tight_layout()
    
    # 保存为PNG文件
    plt.savefig('d:/github/Opt/constructed_matrix.png', dpi=300, bbox_inches='tight')
    print("评分矩阵可视化已保存为 constructed_matrix.png")
    plt.close()  # 关闭图形避免内存泄漏

def save_matrix(matrix, filename):
    """保存矩阵到文件并可视化"""
    matrix.to_csv(f'd:/github/Opt/{filename}')
    print(f"矩阵已保存为 {filename}")
    visualize_matrix(matrix, f'评分矩阵: {filename}')

if __name__ == '__main__':
    # 创建正态分布缺失矩阵
    normal_matrix = create_normal_distribution_matrix()
    save_matrix(normal_matrix, 'normal_missing_matrix.csv')