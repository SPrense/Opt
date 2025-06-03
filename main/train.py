import torch
import os
import pickle 
import wandb 
import pandas as pd 

def evaluate_model(model, test_loader, criterion, device, log_predictions_to_wandb=False, current_iter_num=0, current_epoch_num=0):
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
            # 确保预测和目标形状匹配criterion
            if predictions.shape != ratings_batch.shape and predictions.numel() == ratings_batch.numel():
                 loss = criterion(predictions.squeeze(), ratings_batch.squeeze())
            else:
                 loss = criterion(predictions, ratings_batch)
            total_test_loss += loss.item() * user_ids_batch.size(0) 
            count += user_ids_batch.size(0)

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
    if log_predictions_to_wandb and predictions_for_wandb:
        try:
            data_for_table = [[u, i, p, a] for u,i,p,a in zip(user_ids_for_wandb, item_ids_for_wandb, predictions_for_wandb, actuals_for_wandb)]
            wandb_table = wandb.Table(data=data_for_table, columns=["User ID", "Item ID", "Predicted", "Actual"])
        except Exception as e:
            print(f"创建W&B Table失败: {e}")
            wandb_table = None 

    return avg_test_loss, wandb_table


def train_model_epoch(model, train_loader, test_loader, criterion, optimizer, epoch, total_epochs_this_iteration, 
                      is_checkpoint_epoch, learning_rate, device, current_iteration_num, global_epoch_step, args):
    model.train()
    total_train_loss = 0
    train_count = 0
    current_epoch_gradient_log = [] 

    if not train_loader or len(train_loader.dataset) == 0: 
        print(f"  Iter {current_iteration_num} Epoch [{epoch+1}/{total_epochs_this_iteration}], 训练数据为空，跳过训练。")
        avg_test_loss = 0.0
        wandb_test_preds_table = None
        if test_loader and is_checkpoint_epoch: 
             avg_test_loss, wandb_test_preds_table = evaluate_model(model, test_loader, criterion, device, 
                                                                  log_predictions_to_wandb=True, 
                                                                  current_iter_num=current_iteration_num, 
                                                                  current_epoch_num=epoch+1)
             print(f"  Iter {current_iteration_num} Epoch [{epoch+1}/{total_epochs_this_iteration}], Test Loss: {avg_test_loss:.6f} (训练数据为空)")
             if not args.disable_wandb:
                log_dict = {
                    f"Iter_{current_iteration_num}/AvgTrainLoss_epoch": 0.0,
                    f"Iter_{current_iteration_num}/TestLoss_epoch": avg_test_loss,
                    "global_epoch": global_epoch_step + epoch,
                    "iteration": current_iteration_num,
                    "epoch_in_iteration": epoch + 1
                }
                if wandb_test_preds_table:
                    log_dict[f"Iter_{current_iteration_num}/Epoch_{epoch+1}_TestPredictions"] = wandb_test_preds_table
                wandb.log(log_dict, step=global_epoch_step + epoch) 
        return [], 0.0


    for batch_idx, (user_ids_batch, item_ids_batch, ratings_batch) in enumerate(train_loader):
        user_ids_batch = user_ids_batch.to(device)
        item_ids_batch = item_ids_batch.to(device)
        ratings_batch = ratings_batch.to(device)

        optimizer.zero_grad()
        
        if is_checkpoint_epoch:
            for i in range(user_ids_batch.size(0)):
                user_id_tensor = user_ids_batch[i:i+1] 
                item_id_tensor = item_ids_batch[i:i+1] 
                rating_tensor = ratings_batch[i:i+1]   

                model.zero_grad() 
                single_prediction = model(user_id_tensor, item_id_tensor)
                if single_prediction.shape != rating_tensor.shape and single_prediction.numel() == rating_tensor.numel():
                    single_loss = criterion(single_prediction.squeeze(), rating_tensor.squeeze())
                else:
                    single_loss = criterion(single_prediction, rating_tensor)
                single_loss.backward() 

                user_id = user_id_tensor.item()
                item_id = item_id_tensor.item()
                
                user_grad = torch.zeros(model.user_embeddings.embedding_dim, device=device)
                if model.user_embeddings.weight.grad is not None and \
                   user_id < model.user_embeddings.num_embeddings and \
                   model.user_embeddings.weight.grad[user_id] is not None: 
                    user_grad = model.user_embeddings.weight.grad[user_id].clone().detach()
                
                item_grad = torch.zeros(model.item_embeddings.embedding_dim, device=device)
                if model.item_embeddings.weight.grad is not None and \
                   item_id < model.item_embeddings.num_embeddings and \
                   model.item_embeddings.weight.grad[item_id] is not None: 
                    item_grad = model.item_embeddings.weight.grad[item_id].clone().detach()

                current_epoch_gradient_log.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'user_grad': user_grad.cpu(), 
                    'item_grad': item_grad.cpu(), 
                    'learning_rate': learning_rate 
                })
            
            optimizer.zero_grad() 
            final_batch_predictions = model(user_ids_batch, item_ids_batch)
            final_batch_loss = criterion(final_batch_predictions, ratings_batch)
            final_batch_loss.backward()
            loss_to_accumulate = final_batch_loss
        
        else: 
            predictions = model(user_ids_batch, item_ids_batch)
            loss = criterion(predictions, ratings_batch)
            loss.backward()
            loss_to_accumulate = loss

        optimizer.step()
        total_train_loss += loss_to_accumulate.item() * user_ids_batch.size(0) 
        train_count += user_ids_batch.size(0)


    avg_train_loss = total_train_loss / train_count if train_count > 0 else 0.0
    print(f"  Iter {current_iteration_num} Epoch [{epoch+1}/{total_epochs_this_iteration}], Avg Train Loss: {avg_train_loss:.6f}")
    
    avg_test_loss = 0.0
    wandb_test_preds_table = None
    if test_loader and is_checkpoint_epoch: 
        avg_test_loss, wandb_test_preds_table = evaluate_model(model, test_loader, criterion, device, 
                                                             log_predictions_to_wandb=True,
                                                             current_iter_num=current_iteration_num,
                                                             current_epoch_num=epoch+1)
        print(f"  Iter {current_iteration_num} Epoch [{epoch+1}/{total_epochs_this_iteration}], Test Loss: {avg_test_loss:.6f}")
    
    if not args.disable_wandb and is_checkpoint_epoch: 
        log_dict = {
            f"Iter_{current_iteration_num}/AvgTrainLoss_epoch": avg_train_loss,
            f"Iter_{current_iteration_num}/TestLoss_epoch": avg_test_loss,
            "global_epoch": global_epoch_step + epoch, 
            "iteration": current_iteration_num,
            "epoch_in_iteration": epoch + 1
        }
        if wandb_test_preds_table: 
            log_dict[f"Iter_{current_iteration_num}/Epoch_{epoch+1}_TestPredictions"] = wandb_test_preds_table
        wandb.log(log_dict, step=global_epoch_step + epoch) 

    return current_epoch_gradient_log, avg_train_loss


def run_training_iteration(model, train_loader, test_loader, criterion, optimizer, args, current_iteration_num, global_epoch_offset, device):
    strategy_iter_checkpoint_dir = os.path.join(args.checkpoint_dir_base, f"strategy_{args.active_learning_strategy}", f"iter_{current_iteration_num}")
    strategy_iter_gradient_log_dir = os.path.join(args.gradient_log_dir_base, f"strategy_{args.active_learning_strategy}", f"iter_{current_iteration_num}")
    os.makedirs(strategy_iter_checkpoint_dir, exist_ok=True)
    os.makedirs(strategy_iter_gradient_log_dir, exist_ok=True)
    
    gradient_log_paths_map = {} 

    print(f"--- 开始训练迭代轮次 {current_iteration_num} (共 {args.epochs_per_iteration} epochs) ---")
    if not train_loader or len(train_loader.dataset) == 0: 
        print(f"迭代轮次 {current_iteration_num} 的训练数据为空，跳过此轮训练。")
        if not args.disable_wandb:
            final_global_epoch_this_iter = global_epoch_offset + args.epochs_per_iteration -1
            initial_model_test_loss = 0.0
            wandb_test_table_empty_train = None
            if test_loader: # 即使训练数据为空，也评估一下模型在测试集上的表现
                initial_model_test_loss, wandb_test_table_empty_train = evaluate_model(model, test_loader, criterion, device, log_predictions_to_wandb=True, current_iter_num=current_iteration_num, current_epoch_num=0)

            log_data_empty_train = {
                f"Iter_{current_iteration_num}/AvgTrainLoss_epoch": 0.0,
                f"Iter_{current_iteration_num}/TestLoss_epoch": initial_model_test_loss,
                "global_epoch": final_global_epoch_this_iter, 
                "iteration": current_iteration_num,
                "epoch_in_iteration": args.epochs_per_iteration 
            }
            if wandb_test_table_empty_train:
                 log_data_empty_train[f"Iter_{current_iteration_num}/Epoch_0_TestPredictions_EmptyTrain"] = wandb_test_table_empty_train
            wandb.log(log_data_empty_train, step=final_global_epoch_this_iter)
        return gradient_log_paths_map 

    for epoch in range(args.epochs_per_iteration): 
        is_checkpoint_epoch = (epoch + 1) % args.checkpoint_interval == 0 or (epoch + 1) == args.epochs_per_iteration
        
        epoch_gradient_log, avg_train_loss = train_model_epoch(
            model, train_loader, test_loader, criterion, optimizer, epoch, args.epochs_per_iteration,
            is_checkpoint_epoch, args.learning_rate, device, current_iteration_num, global_epoch_offset, args
        )

        if is_checkpoint_epoch:
            checkpoint_path = os.path.join(strategy_iter_checkpoint_dir, f"model_epoch_{epoch+1}.pt") 
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    已保存checkpoint: {checkpoint_path}")

            if epoch_gradient_log: 
                grad_log_filename = f"checkpoint_epoch_{epoch+1}_grads.pkl"
                grad_log_path = os.path.join(strategy_iter_gradient_log_dir, grad_log_filename) 
                with open(grad_log_path, 'wb') as f:
                    pickle.dump(epoch_gradient_log, f)
                gradient_log_paths_map[epoch + 1] = grad_log_path
                print(f"    已保存梯度日志: {grad_log_path} (包含 {len(epoch_gradient_log)} 条记录)")
    
    print(f"--- 训练迭代轮次 {current_iteration_num} 完成 ---")
    return gradient_log_paths_map