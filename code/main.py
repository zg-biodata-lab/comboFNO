import pandas as pd
import numpy as np
import pickle
import os
import json
import torch
from tqdm import tqdm
import itertools
from data_processing import process_single_combo
from train_eval import train_for_single_cell_line

# --- Set scenario and response type ---
scenario = 'new_combo'; response = 'PERCENTGROWTHNOTZ'
# scenario = 'new_combo'; response = 'PERCENTGROWTH'
# scenario = 'new_drug'; response = 'PERCENTGROWTHNOTZ'
# scenario = 'new_drug'; response = 'PERCENTGROWTH'

if scenario == 'new_combo':
    CONFIG = {
        "seed": 40, 
        "standard_bounds": [[-12.0, -4.0], [-12.0, -4.0]],
        "data": {
            "response_path": os.path.join("..", "data", f'data_matrix_{response}.pickle'),
            "feature_path": os.path.join("..", "data", 'NCI-ALMANAC_drug_molformer_embeddings.pt')
        },
        "model": {
            "fno_modes": 8, "fno_width": 12, "fno_n_layers": 4, "grid": 16, "decoder_hidden_dims": [32,16], "feature_dim": 768, 
            "dropout_rate1": 0.1, "dropout_rate2": 0.1, "lifting": [32]
        },
        "training": {
            "epochs": 200, "batch_size": 32, "learning_rate": 2e-4, "weight_decay": 1e-4, "patience": 30
        },
        "experiment": { "target_cell_line": None, "type": scenario, }
    }
if scenario == 'new_drug':
    CONFIG = {
        "seed": 40, 
        "standard_bounds": [[-12.0, -4.0], [-12.0, -4.0]],
        "data": {
            "response_path": os.path.join("..", "data", f'data_matrix_{response}.pickle'),
            "feature_path": os.path.join("..", "data", 'NCI-ALMANAC_drug_molformer_embeddings.pt')
        },
        "model": {
            "fno_modes": 4, "fno_width": 36, "fno_n_layers": 4, "grid": 8, "decoder_hidden_dims": [128,64], "feature_dim": 768, 
            "dropout_rate1": 0.1, "dropout_rate2": 0.1, "lifting": [256]
        },
        "training": {
            "epochs": 200, "batch_size": 32, "learning_rate": 2e-4, "weight_decay": 1e-4, "patience": 30
        },
        "experiment": { "target_cell_line": None, "type": scenario, }
    }   

# --- Result save path ---
experiment_name = CONFIG['experiment']['type']
output_dir = os.path.join("..", "result", experiment_name)
os.makedirs(output_dir, exist_ok=True)

# --- Data loading and preprocessing ---
try:
    with open(CONFIG['data']['response_path'], 'rb') as f:
        raw_data_dict = pickle.load(f)
    drug_features_dict = torch.load(CONFIG['data']['feature_path'], map_location='cpu', weights_only=True)
    print("[INFO] Loading raw data and features... Done.")
except FileNotFoundError as e:
    print(f"Error: Data file not found:{e}")
else:
    all_samples_for_splitting = []
    skipped_combos = []
    
    for cell_line, combos in itertools.islice(raw_data_dict.items(), 0, None):
        for combo_name, raw_matrix in combos.items():
            try:
                drug1, drug2 = combo_name.split('_'); d1_clean, d2_clean = drug1.strip(), drug2.strip()
                if d1_clean not in drug_features_dict or d2_clean not in drug_features_dict:
                    skipped_combos.append(f"cellname: {cell_line}, combination: {combo_name}")
                    continue 
            except ValueError:
                skipped_combos.append(f"cellname: {cell_line}, Combination name format error: {combo_name}")
                continue
                
            processed_data = process_single_combo(np.array(raw_matrix), CONFIG)
            if processed_data:
                original_sample = processed_data.copy()
                original_sample['cell_line'] = cell_line
                original_sample['combo_name'] = f"{d1_clean}_{d2_clean}"
                all_samples_for_splitting.append(original_sample)
                
                flipped_sample = processed_data.copy()
                flipped_sample['cell_line'] = cell_line
                flipped_sample['combo_name'] = f"{d2_clean}_{d1_clean}"
                flipped_sample['sparse_coords_log'] = flipped_sample['sparse_coords_log'][:, [1, 0]]
                all_samples_for_splitting.append(flipped_sample)    
    print("[INFO] Data preprocessing...Done.")

    if skipped_combos:
        print("\n" + "="*50)
        print(f"[warning] The following {len(skipped_combos)} drug combinations were skipped due to missing corresponding features or incorrect name format:")
        for item in skipped_combos[:10]:
            print(f"  - {item}")
        if len(skipped_combos) > 10:
            print(f"  ... (And other {len(skipped_combos) - 10})")
        print("="*50)
    else:
        print("[INFO] Feature alignment: 100% of drug combination features matched.")


    # --- Start training cycle ---
    all_cell_lines = list(raw_data_dict.keys())
    all_test_metrics = []

    for i, cell_line in enumerate(all_cell_lines):
        test_metrics = train_for_single_cell_line(
                                    cell_line_name=cell_line,
                                    cell_num=i,
                                    base_config=CONFIG,
                                    all_samples=all_samples_for_splitting,
                                    drug_features=drug_features_dict,
                                    base_output_dir=output_dir,
                                    scenario = scenario,
                                    response = response
                                    )
        test_metrics['CellLine'] = cell_line
        all_test_metrics.append(test_metrics)

    # --- Save the result ---
    print(f"\nAll experiments have been completed!")
    summary_df = pd.DataFrame(all_test_metrics)
    if 'CellLine' in summary_df.columns:
        cols = ['CellLine'] + [col for col in summary_df.columns if col != 'CellLine']
        summary_df = summary_df[cols]
    summary_filename = f"fno_{response}_result.csv"
    summary_output_path = os.path.join(output_dir, summary_filename)
    summary_df.to_csv(summary_output_path, index=False)
    print(f"The result has been saved to: {summary_output_path}")