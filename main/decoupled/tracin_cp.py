import torch
import torch.nn as nn
import os
import pickle
from collections import defaultdict
import wandb # 确保导入wandb，如果评估函数在这里面用到了

# 将 evaluate_model 移到这里
def evaluate_model_on_test_set(model, test_loader, criterion, device, 
                               log_predictions_to_wandb=False, current_iter_num=0, current_epoch_num=0):
    """在测试集上评估模型，并可选择记录到W&B"""
    model.eval() 
    total_test_loss = 0
    count = 0
    
    predictions_for_wandb = []
    actuals_for_wandb = []
    user_ids_for_wandb = []
    item_ids_for_wandb = []

    if not test_loader or len(test_loader.dataset) == 0: 
        return 0.0, None 

    with torch.no_grad(): 
        for user_ids_batch, item_ids_batch, ratings_batch in test_loader:
            user_ids_list = user_ids_batch.squeeze().tolist()
            item_ids_list = item_ids_batch.squeeze().tolist()
            ratings_list = ratings_batch.squeeze().tolist()

            user_ids_batch = user_ids_batch.to(device)
            item_ids_batch = item_ids_batch.to(device)
            ratings_batch = ratings_batch.to(device)

            predictions = model(user_ids_batch, item_ids_batch)
            
            loss_preds = predictions
            loss_targets = ratings_batch
            if predictions.shape != ratings_batch.shape:
                if predictions.numel() == ratings_batch.numel(): # 确保元素数量一致才能squeeze
                    loss_preds = predictions.squeeze()
                    loss_targets = ratings_batch.squeeze()
                else:
                    print(f"警告 (evaluate_model): 预测和目标形状及元素数量均不匹配。Pred: {predictions.shape}, Target: {ratings_batch.shape}")
                    continue 
            
            try:
                loss = criterion(loss_preds, loss_targets)
                total_test_loss += loss.item() * user_ids_batch.size(0) 
                count += user_ids_batch.size(0)
            except RuntimeError as e:
                print(f"错误 (evaluate_model) 计算损失时: {e}")
                print(f"  Pred shape: {loss_preds.shape}, Target shape: {loss_targets.shape}")
                continue


            if log_predictions_to_wandb:
                preds_cpu = predictions.squeeze().cpu().tolist()
                
                if not isinstance(user_ids_list, list): user_ids_list = [user_ids_list]
                if not isinstance(item_ids_list, list): item_ids_list = [item_ids_list]
                if not isinstance(preds_cpu, list): preds_cpu = [preds_cpu]
                if not isinstance(ratings_list, list): ratings_list = [ratings_list]


                user_ids_for_wandb.extend(user_ids_list)
                item_ids_for_wandb.extend(item_ids_list)
                predictions_for_wandb.extend(preds_cpu)
                actuals_for_wandb.extend(ratings_list)
    
    avg_test_loss = total_test_loss / count if count > 0 else 0.0
    
    wandb_table = None
    if log_predictions_to_wandb and predictions_for_wandb and wandb.run is not None: 
        try:
            min_len = min(len(user_ids_for_wandb), len(item_ids_for_wandb), len(predictions_for_wandb), len(actuals_for_wandb))
            data_for_table = [[user_ids_for_wandb[i], item_ids_for_wandb[i], predictions_for_wandb[i], actuals_for_wandb[i]] for i in range(min_len)]
            if data_for_table: 
                 table_name = f"Iter_{current_iter_num}_Epoch_{current_epoch_num}_TestSetPredictions" # 确保表名唯一
                 wandb_table = wandb.Table(data=data_for_table, columns=["User ID", "Item ID", "Predicted", "Actual"])
        except Exception as e:
            print(f"创建W&B Table失败: {e}")
            wandb_table = None 

    return avg_test_loss, wandb_table


class TracInCP_Decoupled:
    def __init__(self, model_class_constructor, model_args, task_type,
                 current_checkpoint_dir, current_gradient_log_paths_map, criterion):
        self.model_class_constructor = model_class_constructor
        self.model_args = model_args 
        self.task_type = task_type
        self.current_checkpoint_dir = current_checkpoint_dir
        self.current_gradient_log_paths_map = current_gradient_log_paths_map
        self.criterion = criterion

        if self.task_type not in ['regression', 'classification']:
            raise ValueError(f"不支持的任务类型: {self.task_type}")

    def _get_single_instance_gradients(self, model_at_checkpoint, user_tensor, item_tensor, rating_tensor):
        device = next(model_at_checkpoint.parameters()).device 
        user_tensor = user_tensor.to(device)
        item_tensor = item_tensor.to(device)
        rating_tensor = rating_tensor.to(device)
        
        model_at_checkpoint.zero_grad()
        model_at_checkpoint.eval() 

        prediction = model_at_checkpoint(user_tensor, item_tensor)
        
        loss_pred = prediction
        loss_target = rating_tensor
        if prediction.shape != rating_tensor.shape:
            if prediction.numel() == rating_tensor.numel():
                loss_pred = prediction.squeeze()
                loss_target = rating_tensor.squeeze()
            else:
                print(f"严重警告 (TracInCP _get_single_instance_gradients): 预测和目标形状及元素数量均不匹配。Pred: {prediction.shape}, Target: {rating_tensor.shape}")
                return torch.zeros_like(model_at_checkpoint.user_embeddings.weight[0], device=device), \
                       torch.zeros_like(model_at_checkpoint.item_embeddings.weight[0], device=device)
        
        try:
            loss = self.criterion(loss_pred, loss_target)
            loss.backward()
        except RuntimeError as e:
            print(f"错误 (TracInCP _get_single_instance_gradients) 计算损失或反向传播时: {e}")
            print(f"  Pred shape: {loss_pred.shape}, Target shape: {loss_target.shape}")
            return torch.zeros_like(model_at_checkpoint.user_embeddings.weight[0], device=device), \
                   torch.zeros_like(model_at_checkpoint.item_embeddings.weight[0], device=device)


        user_id = user_tensor.item()
        item_id = item_tensor.item()

        grad_user_params = torch.zeros_like(model_at_checkpoint.user_embeddings.weight[0], device=device)
        if model_at_checkpoint.user_embeddings.weight.grad is not None and \
           user_id < model_at_checkpoint.user_embeddings.num_embeddings and \
           model_at_checkpoint.user_embeddings.weight.grad[user_id] is not None:
            grad_user_params = model_at_checkpoint.user_embeddings.weight.grad[user_id].clone().detach()

        grad_item_params = torch.zeros_like(model_at_checkpoint.item_embeddings.weight[0], device=device)
        if model_at_checkpoint.item_embeddings.weight.grad is not None and \
           item_id < model_at_checkpoint.item_embeddings.num_embeddings and \
           model_at_checkpoint.item_embeddings.weight.grad[item_id] is not None:
            grad_item_params = model_at_checkpoint.item_embeddings.weight.grad[item_id].clone().detach()
        
        return grad_user_params, grad_item_params


    def compute_cumulative_influences_on_test_set(self, test_loader): 
        total_user_influences = defaultdict(float) 
        total_item_influences = defaultdict(float) 
        
        # 新增：用于计算单位影响值的计数器
        user_dot_product_counts = defaultdict(int)
        item_dot_product_counts = defaultdict(int)
        
        device_for_loading = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(self, 'model_args') and self.model_args:
            try:
                if not all(k in self.model_args for k in ['num_users', 'num_items', 'embedding_dim']):
                    print("警告 (TracInCP): model_args 不完整。")
                elif self.model_args['num_users'] > 0 and self.model_args['num_items'] > 0 : 
                    temp_model_for_device_check = self.model_class_constructor(**self.model_args)
                    device_for_loading = next(temp_model_for_device_check.parameters()).device
                    del temp_model_for_device_check
            except Exception as e:
                print(f"警告 (TracInCP): 尝试从模型推断设备失败: {e}。将使用默认设备 {device_for_loading}。")
                pass 

        if not self.current_gradient_log_paths_map: 
            print("    警告: (TracInCP) 没有梯度日志路径图提供。无法计算影响。")
            return defaultdict(float), defaultdict(float) # 返回空的defaultdict

        if not test_loader or len(test_loader.dataset) == 0:
            print("    警告: (TracInCP) 测试集为空，无法计算对测试集误差的影响。影响力将为0。")
            return defaultdict(float), defaultdict(float)

        sorted_checkpoint_epochs = sorted(self.current_gradient_log_paths_map.keys())
        
        print("  TracInCP: 正在为每个checkpoint和每个测试样本计算梯度并累加影响...")
        for epoch_num in sorted_checkpoint_epochs:
            checkpoint_model_path = os.path.join(self.current_checkpoint_dir, f"model_epoch_{epoch_num}.pt")
            gradient_log_path = self.current_gradient_log_paths_map[epoch_num]

            if not os.path.exists(checkpoint_model_path) or not os.path.exists(gradient_log_path):
                continue
            
            if self.model_args.get('num_users', 0) <= 0 or self.model_args.get('num_items', 0) <= 0:
                continue
                
            current_model = self.model_class_constructor(**self.model_args)
            current_model.load_state_dict(torch.load(checkpoint_model_path, map_location=device_for_loading))
            current_model.to(device_for_loading) 

            try:
                with open(gradient_log_path, 'rb') as f:
                    training_gradients_log = pickle.load(f) 
            except Exception as e:
                continue

            if not training_gradients_log:
                continue
            
            for test_user_batch, test_item_batch, test_rating_batch in test_loader:
                for i in range(test_user_batch.size(0)): 
                    test_u_tensor = test_user_batch[i:i+1]
                    test_i_tensor = test_item_batch[i:i+1]
                    test_r_tensor = test_rating_batch[i:i+1]

                    grad_test_user_for_sample, grad_test_item_for_sample = self._get_single_instance_gradients(
                        current_model,
                        test_u_tensor,
                        test_i_tensor,
                        test_r_tensor
                    )

                    for train_grad_info in training_gradients_log:
                        train_user_id = train_grad_info['user_id']
                        train_item_id = train_grad_info['item_id']
                        grad_train_user = train_grad_info['user_grad'].to(device_for_loading) 
                        grad_train_item = train_grad_info['item_grad'].to(device_for_loading)
                        learning_rate = train_grad_info['learning_rate']

                        user_dot_product = torch.dot(grad_train_user, grad_test_user_for_sample)
                        total_user_influences[train_user_id] += learning_rate * user_dot_product.item()
                        user_dot_product_counts[train_user_id] += 1 # 每次点积计数加1
                        
                        item_dot_product = torch.dot(grad_train_item, grad_test_item_for_sample)
                        total_item_influences[train_item_id] += learning_rate * item_dot_product.item()
                        item_dot_product_counts[train_item_id] += 1 # 每次点积计数加1
        
        # 计算单位影响值
        unit_user_influences = defaultdict(float)
        for user_id, total_influence in total_user_influences.items():
            if user_dot_product_counts[user_id] > 0:
                unit_user_influences[user_id] = total_influence / user_dot_product_counts[user_id]
        
        unit_item_influences = defaultdict(float)
        for item_id, total_influence in total_item_influences.items():
            if item_dot_product_counts[item_id] > 0:
                unit_item_influences[item_id] = total_influence / item_dot_product_counts[item_id]
                
        if not unit_user_influences and not unit_item_influences and sorted_checkpoint_epochs:
            print("    警告: (TracInCP) 未计算出任何单位影响分数。")
            
        return unit_user_influences, unit_item_influences # 返回单位影响值