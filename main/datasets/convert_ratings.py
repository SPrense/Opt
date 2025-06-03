import pandas as pd
import numpy as np

def convert_ratings_to_matrix(input_file, output_file):
    """
    将 sub_ratings.dat 文件转换为矩阵格式的 CSV 文件
    
    Args:
        input_file (str): 输入文件路径 (sub_ratings.dat)
        output_file (str): 输出文件路径 (转换后的csv文件)
    """
    # 读取原始数据
    ratings = []
    with open(input_file, 'r') as f:
        for line in f:
            # 分割每行数据，格式为 "user_id::item_id::rating"
            user_id, item_id, rating = line.strip().split('::')
            ratings.append({
                'user_id': int(user_id),
                'item_id': int(item_id),
                'rating': float(rating)
            })
    
    # 转换为 DataFrame
    df = pd.DataFrame(ratings)
    
    # 获取用户数和物品数的最大值
    max_user_id = df['user_id'].max()
    max_item_id = df['item_id'].max()
    
    # 创建空矩阵
    matrix = pd.DataFrame(
        np.nan,  # 使用 NaN 表示缺失值
        index=range(1, max_user_id + 1),
        columns=range(1, max_item_id + 1)
    )
    
    # 填充评分
    for _, row in df.iterrows():
        matrix.loc[row['user_id'], row['item_id']] = row['rating']
    
    # 保存为 CSV 文件
    matrix.to_csv(output_file)
    
    print(f"转换完成！")
    print(f"矩阵大小: {matrix.shape}")
    print(f"非空评分数量: {df.shape[0]}")
    print(f"已保存至: {output_file}")

if __name__ == "__main__":
    input_file = "sub_ratings.dat"
    output_file = "converted_ratings.csv"
    
    convert_ratings_to_matrix(input_file, output_file)