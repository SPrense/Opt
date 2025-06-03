import random
import numpy as np
from collections import Counter

def get_random_collection_points(num_points_to_collect, num_total_users, num_total_items, existing_ratings_set):
    """
    随机选择未评分的用户-物品对。
    """
    collection_points = []
    
    if num_total_users == 0 or num_total_items == 0:
        print("  警告 (Random Baseline): 用户总数或物品总数为0，无法生成随机点。")
        return collection_points

    # 为了避免无限循环（如果矩阵非常密集），可以设置一个尝试上限
    # 启发式上限，例如所需点数的100倍，或者总可能点数的一小部分
    max_tries_heuristic = num_points_to_collect * 100
    max_tries_total_points_fraction = (num_total_users * num_total_items) // 10
    max_tries = min(max_tries_heuristic, max_tries_total_points_fraction)
    max_tries = max(max_tries, num_points_to_collect * 20) # 确保至少有足够的尝试次数

    # 如果可能未评分的点很少，直接从这些点中抽样
    num_possible_pairs = num_total_users * num_total_items
    if num_possible_pairs - len(existing_ratings_set) < num_points_to_collect * 5 and num_possible_pairs < 2_000_000 : # 调整阈值
        all_possible_pairs = set((u, i) for u in range(num_total_users) for i in range(num_total_items))
        unrated_pairs = list(all_possible_pairs - existing_ratings_set)
        random.shuffle(unrated_pairs)
        collection_points = unrated_pairs[:num_points_to_collect]
    else: # 如果矩阵太大或未评分点很多，则随机采样
        tries = 0
        # 使用集合来快速检查重复添加
        collected_set_for_this_func = set()
        while len(collection_points) < num_points_to_collect and tries < max_tries:
            tries += 1
            rand_user = random.randint(0, num_total_users - 1)
            rand_item = random.randint(0, num_total_items - 1)
            if (rand_user, rand_item) not in existing_ratings_set:
                if (rand_user, rand_item) not in collected_set_for_this_func: 
                     collection_points.append((rand_user, rand_item))
                     collected_set_for_this_func.add((rand_user,rand_item)) # 添加到临时集合
        if tries >= max_tries and len(collection_points) < num_points_to_collect:
            print(f"  警告 (Random Baseline): 达到最大尝试次数 {max_tries}，但只收集到 {len(collection_points)}/{num_points_to_collect} 个点。")

    print(f"  随机策略选择了 {len(collection_points)} 个采集点。")
    return collection_points


def get_active_user_collection_points(num_points_to_collect, current_dataset_tuples, 
                                      num_total_users, num_total_items, existing_ratings_set):
    collection_points = []
    if num_total_users == 0 or num_total_items == 0:
        print("  警告 (ActiveUser Baseline): 用户总数或物品总数为0。")
        return collection_points

    if not current_dataset_tuples: 
        print("  警告 (ActiveUser Baseline): 当前训练数据为空，回退到随机选择。")
        return get_random_collection_points(num_points_to_collect, num_total_users, num_total_items, existing_ratings_set)

    user_activity = Counter(u for u, i, r in current_dataset_tuples)
    sorted_active_users = [u for u, count in user_activity.most_common() if u < num_total_users] # 确保用户ID有效

    all_users_set = set(range(num_total_users))
    active_users_set = set(sorted_active_users)
    inactive_users = list(all_users_set - active_users_set)
    random.shuffle(inactive_users) 
    
    potential_users_ordered = sorted_active_users + inactive_users
    collected_set_for_this_func = set()


    for user_id in potential_users_ordered:
        if len(collection_points) >= num_points_to_collect:
            break
        
        possible_items_for_user = list(range(num_total_items))
        random.shuffle(possible_items_for_user) 
        
        items_collected_for_this_user = 0
        max_items_per_user_heuristic = max(1, num_points_to_collect // (len(potential_users_ordered) // 2 + 1) +1) # 启发式限制每个用户选太多

        for item_id in possible_items_for_user:
            if (user_id, item_id) not in existing_ratings_set:
                if (user_id, item_id) not in collected_set_for_this_func:
                    collection_points.append((user_id, item_id))
                    collected_set_for_this_func.add((user_id, item_id))
                    items_collected_for_this_user +=1
                    if len(collection_points) >= num_points_to_collect or items_collected_for_this_user >= max_items_per_user_heuristic:
                        break
        if len(collection_points) >= num_points_to_collect:
            break
            
    print(f"  活跃用户策略选择了 {len(collection_points)} 个采集点。")
    return collection_points


def get_popular_item_collection_points(num_points_to_collect, current_dataset_tuples,
                                       num_total_users, num_total_items, existing_ratings_set):
    collection_points = []
    if num_total_users == 0 or num_total_items == 0:
        print("  警告 (PopularItem Baseline): 用户总数或物品总数为0。")
        return collection_points

    if not current_dataset_tuples:
        print("  警告 (PopularItem Baseline): 当前训练数据为空，回退到随机选择。")
        return get_random_collection_points(num_points_to_collect, num_total_users, num_total_items, existing_ratings_set)

    item_popularity = Counter(i for u, i, r in current_dataset_tuples)
    sorted_popular_items = [i for i, count in item_popularity.most_common() if i < num_total_items] # 确保物品ID有效

    all_items_set = set(range(num_total_items))
    popular_items_set = set(sorted_popular_items)
    unpopular_items = list(all_items_set - popular_items_set)
    random.shuffle(unpopular_items)

    potential_items_ordered = sorted_popular_items + unpopular_items
    collected_set_for_this_func = set()


    for item_id in potential_items_ordered:
        if len(collection_points) >= num_points_to_collect:
            break
        
        possible_users_for_item = list(range(num_total_users))
        random.shuffle(possible_users_for_item)
        
        users_collected_for_this_item = 0
        max_users_per_item_heuristic = max(1, num_points_to_collect // (len(potential_items_ordered) // 2 + 1) + 1)


        for user_id in possible_users_for_item:
            if (user_id, item_id) not in existing_ratings_set:
                 if (user_id, item_id) not in collected_set_for_this_func:
                    collection_points.append((user_id, item_id))
                    collected_set_for_this_func.add((user_id,item_id))
                    users_collected_for_this_item +=1
                    if len(collection_points) >= num_points_to_collect or users_collected_for_this_item >= max_users_per_item_heuristic:
                        break
        if len(collection_points) >= num_points_to_collect:
            break

    print(f"  热门物品策略选择了 {len(collection_points)} 个采集点。")
    return collection_points