import os
import torch
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from general_utils import set_seed, sanitize_filename
from datasets import DrugResponsePointDataset
from fno_model import FNOPointwiseModel
from data_processing import collate_fn_pad, split_points, split_data_for_cell_line

def train_for_single_cell_line(cell_line_name, cell_num, base_config, all_samples, drug_features, base_output_dir, scenario, response):
    """
    Perform a complete training, evaluation, and model preservation process for a single cell line
    """
    print(f"\n{'-'*50}\n[TASK] Evaluating Cell Line: {cell_num + 1}/60  {cell_line_name}\n{'-'*50}")
    
    task_config = base_config.copy()
    task_config['experiment']['target_cell_line'] = cell_line_name 
    set_seed(task_config['seed'])
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    safe_cell_line_name = sanitize_filename(cell_line_name)
    model_save_dir = os.path.join(base_output_dir, "model", response)
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, f"{safe_cell_line_name}_model.pth")
    
    # --- Dataset splitting ---
    train_list, val_list, test_list, key_for_current_cell = split_data_for_cell_line(all_samples, task_config['experiment'])
    
    if not train_list or not val_list or not test_list:
        print(f"The data of cell line {cell_line_name} is insufficient for three-way partitioning and has been skipped.")
        return None
    if scenario == 'new_combo':
        for s in train_list: s['sparse_y'] = s['sparse_y'] / 100.0
        for s in val_list: s['sparse_y'] = s['sparse_y'] / 100.0
        for s in test_list: s['sparse_y'] = s['sparse_y'] / 100.0
    train_dataset = DrugResponsePointDataset(train_list, drug_features)
    val_dataset = DrugResponsePointDataset(val_list, drug_features)
    test_dataset = DrugResponsePointDataset(test_list, drug_features)
    
    train_loader = DataLoader(train_dataset, batch_size=task_config['training']['batch_size'], shuffle=True, collate_fn=collate_fn_pad)
    val_loader = DataLoader(val_dataset, batch_size=task_config['training']['batch_size'], collate_fn=collate_fn_pad)
    test_loader = DataLoader(test_dataset, batch_size=task_config['training']['batch_size'], collate_fn=collate_fn_pad)
    

    
#     # --- Model initialization ---
#     model = FNOPointwiseModel(task_config).to(DEVICE)
#     criterion = torch.nn.MSELoss() 
#     optimizer = torch.optim.AdamW(model.parameters(), lr=task_config['training']['learning_rate'], weight_decay=task_config['training']['weight_decay'])
# #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
# #     optimizer, 
# #     mode='min', 
# #     factor=0.5,
# #     patience=5,
# #     verbose=True,
# #     min_lr=1e-6
# #     )
#     print("Checking Optimizer Defaults:")
#     print(optimizer.defaults) 
#     print(f"Current Weight Decay: {optimizer.param_groups[0]['weight_decay']}")
#     # --- Training and Early Stopping ---
#     print("Start training the model...")
#     patience = task_config['training'].get("patience", 10)
#     best_val_loss = float('inf')
#     epochs_no_improve = 0

#     for epoch in range(task_config['training']['epochs']):
#         model.train()
#         total_train_loss, train_batches = 0, 0
#         for batch in train_loader:
#             if batch is None: continue
#             features, coords, bounds, targets, mask = (batch['drug_features'].to(DEVICE), batch['sparse_coords_log'].to(DEVICE), batch['log_coord_bounds'].to(DEVICE), batch['sparse_y'].to(DEVICE), batch['attention_mask'].to(DEVICE))
#             optimizer.zero_grad()
#             outputs = model(features, coords, bounds)
#             outputs_flat = outputs.view(-1); targets_flat = targets.view(-1); mask_flat = mask.view(-1)
#             loss = criterion(outputs_flat[mask_flat], targets_flat[mask_flat])
#             loss.backward(); optimizer.step()
#             total_train_loss += loss.item(); train_batches += 1
#         avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0

#         model.eval()
#         total_val_loss, val_batches = 0, 0
#         with torch.no_grad():
#             for batch in val_loader:
#                 if batch is None: continue
#                 features, coords, bounds, targets, mask = (batch['drug_features'].to(DEVICE), batch['sparse_coords_log'].to(DEVICE), batch['log_coord_bounds'].to(DEVICE), batch['sparse_y'].to(DEVICE), batch['attention_mask'].to(DEVICE))
#                 outputs = model(features, coords, bounds)
#                 outputs_flat = outputs.view(-1); targets_flat = targets.view(-1); mask_flat = mask.view(-1)
#                 loss = criterion(outputs_flat[mask_flat], targets_flat[mask_flat])
#                 total_val_loss += loss.item(); val_batches += 1
#         avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0

#         test_mse_sum = 0.0
#         test_total_points = 0

#         model.eval()
#         with torch.no_grad():
#             for batch in test_loader:
#                 if batch is None: continue
#                 features = batch['drug_features'].to(DEVICE)
#                 coords = batch['sparse_coords_log'].to(DEVICE)
#                 bounds = batch['log_coord_bounds'].to(DEVICE)
#                 targets = batch['sparse_y'].to(DEVICE)
#                 mask = batch['attention_mask'].to(DEVICE)

#                 outputs = model(features, coords, bounds)
#                 #pred_real = outputs * 100.0
#                 pred_real = outputs
#                 pred_flat = pred_real.view(-1)
#                 targets_flat = targets.view(-1)
#                 mask_flat = mask.view(-1)

#                 valid_preds = pred_flat[mask_flat]
#                 valid_targets = targets_flat[mask_flat]
#                 # valid_targets 应该是原始值 (0-100)。如果 Dataset 里 test set 没除以 100，那就是对的。
#                 test_mse_sum += torch.sum((valid_preds - valid_targets) ** 2).item()
#                 test_total_points += valid_targets.numel()
#         rmse_test_loss = np.sqrt(test_mse_sum / test_total_points) if test_total_points > 0 else 0.0
#         total_test_loss, test_batches = 0, 0

#         model.eval()
#         if test_loader:
#             with torch.no_grad():
#                 for batch in test_loader:
#                     if batch is None: continue
#                     features, coords, bounds, targets, mask = (batch['drug_features'].to(DEVICE), batch['sparse_coords_log'].to(DEVICE), batch['log_coord_bounds'].to(DEVICE), batch['sparse_y'].to(DEVICE), batch['attention_mask'].to(DEVICE))
#                     outputs = model(features, coords, bounds)
#                     #outputs = outputs * 100.0
#                     outputs_flat = outputs.view(-1); targets_flat = targets.view(-1); mask_flat = mask.view(-1)
#                     loss = criterion(outputs_flat[mask_flat], targets_flat[mask_flat])
#                     total_test_loss += loss.item(); test_batches += 1
#         avg_test_loss = total_test_loss / test_batches if test_batches > 0 else 0

#         print(f"Epoch {epoch+1:03d}/{task_config['training']['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test RMSE: {rmse_test_loss:.4f}")
#         #scheduler.step(avg_val_loss)
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             epochs_no_improve = 0
#             torch.save(model.state_dict(), best_model_path)
#         else:
#             epochs_no_improve += 1

#         if epochs_no_improve >= patience:
#             print(f"Epoch {epoch+1:03d}: The verification loss has not improved for {patience} consecutive epochs, triggering an early stop.")
#             break

#     print(f"Model training completed. The best model has been saved to: {best_model_path}")

    

    # --- Model Evaluation ---   
    final_model = FNOPointwiseModel(task_config).to(DEVICE)
    if not os.path.exists(best_model_path):
        print(f"Warning: Model file {best_model_path} not found, unable to evaluate.")
        return None
    
    final_model.load_state_dict(torch.load(best_model_path, weights_only=False, map_location=DEVICE), strict=False)
    final_model.eval()
    print("[INFO] Loading trained comboFNO model...Done")
    
    def evaluate(loader):
        true_vals, pred_vals = [], []
        with torch.no_grad():
            for batch in loader:
                if batch is None: continue
                features, coords, bounds, targets, mask = (batch['drug_features'].to(DEVICE), batch['sparse_coords_log'].to(DEVICE), batch['log_coord_bounds'].to(DEVICE), batch['sparse_y'], batch['attention_mask'])
                outputs = final_model(features, coords, bounds).cpu()
                outputs_flat = outputs.view(-1); targets_flat = targets.view(-1); mask_flat = mask.view(-1)
                true_vals.append(targets_flat[mask_flat].numpy()); pred_vals.append(outputs_flat[mask_flat].numpy())
        if not true_vals: return np.array([]), np.array([])
        return np.concatenate(true_vals), np.concatenate(pred_vals)
    
    true_test, pred_test = evaluate(test_loader)
    if scenario == 'new_combo':
        true_test = true_test * 100; pred_test = pred_test * 100
    
    # --- Calculate the final metrics ---
    r2 = r2_score(true_test, pred_test)
    rmse = np.sqrt(mean_squared_error(true_test, pred_test))
    r_pearson, p_pearson = pearsonr(true_test, pred_test)
    r_spearman, p_spearman = spearmanr(true_test, pred_test)
    print(f"[RESULT] Performance for {cell_line_name}:")
    print(f"      pearson: {np.round(float(r_pearson),3)}, spearman: {np.round(float(r_spearman),3)}, rmse: {np.round(float(rmse),3)}, r2:{np.round(float(r2),3)}")

    test_metrics = {
        "R2": float(r2), 
        "RMSE": float(rmse), 
        "Pearson": float(r_pearson), 
        "Spearman": float(r_spearman)
    }        
    return test_metrics