import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import random

def process_single_combo(raw_matrix, config):
    """
    Process the raw response matrix of a single drug combination
    raw_matrix: Raw matrix (columns: drug A concentration, drug B concentration, growth response)
    config: Configuration dictionary
    return: The processed dictionary (logarithmic concentration, response value, boundary) or None
    """
    if raw_matrix.shape[0] < 3: return None
    raw_conc_A, raw_conc_B, growth = raw_matrix[:, 0], raw_matrix[:, 1], raw_matrix[:, 2]
    nonzero_A = raw_conc_A[raw_conc_A > 0]; nonzero_B = raw_conc_B[raw_conc_B > 0]
    if nonzero_A.size == 0 or nonzero_B.size == 0: return None
    min_nonzero_A, min_nonzero_B = np.min(nonzero_A), np.min(nonzero_B)
    replace_A, replace_B = min_nonzero_A / 100.0, min_nonzero_B / 100.0
    processed_A = np.where(raw_conc_A == 0, replace_A, raw_conc_A)
    processed_B = np.where(raw_conc_B == 0, replace_B, raw_conc_B)
    log_A, log_B = np.log10(processed_A), np.log10(processed_B)
    sparse_coords_log = np.vstack((log_A, log_B)).T
    return {
        "sparse_coords_log": sparse_coords_log,
        "sparse_y": growth,
        "log_coord_bounds": np.array(config['standard_bounds'])
    }

def collate_fn_pad(batch):
    """
    Batch processing function: padding variable-length sequences (adapting to samples with different numbers of data points)
    batch: batch of samples
    return: The padded batch data or None
    """  
    batch = [item for item in batch if item is not None]
    if not batch: return None
    max_len = max(item['sparse_y'].shape[0] for item in batch)
    padded_batch = []
    for item in batch:
        current_len = item['sparse_y'].shape[0]; pad_len = max_len - current_len; attention_mask = torch.ones(current_len)
        if pad_len > 0:
            item['sparse_y'] = F.pad(item['sparse_y'], (0, pad_len), 'constant', 0)
            item['sparse_coords_log'] = F.pad(item['sparse_coords_log'], (0, 0, 0, pad_len), 'constant', 0)
            attention_mask = F.pad(attention_mask, (0, pad_len), 'constant', 0)
        item['attention_mask'] = attention_mask.bool(); padded_batch.append(item)
    return torch.utils.data.dataloader.default_collate(padded_batch)

def split_points(sample):
    """
    Split the sample into single-drug data and combination data based on concentration coordinates
    sample: Drug combination
    return: Mono_data, combo_data of each combination
    """
    coords = sample['sparse_coords_log']; responses = sample['sparse_y']
    min_c0 = np.min(coords[:, 0]); min_c1 = np.min(coords[:, 1]); mono_mask = (coords[:, 0] == min_c0) | (coords[:, 1] == min_c1)
    combo_mask = ~mono_mask; mono_data, combo_data = None, None
    if np.any(mono_mask):
        mono_data = sample.copy(); mono_data['sparse_coords_log'] = coords[mono_mask]; mono_data['sparse_y'] = responses[mono_mask]
    if np.any(combo_mask):
        combo_data = sample.copy(); combo_data['sparse_coords_log'] = coords[combo_mask]; combo_data['sparse_y'] = responses[combo_mask]
    return mono_data, combo_data

def split_data_for_cell_line(full_dataset, config):
    cell_line_name = config['target_cell_line']; experiment_type = config['type']
    cell_specific_samples = [s for s in full_dataset if s['cell_line'] == cell_line_name]
    if not cell_specific_samples: return [], [], []
    combo_to_samples_map = defaultdict(list); drugs_in_cell_line = set()
    for sample in cell_specific_samples:
        try:
            drug1, drug2 = sample['combo_name'].split('_'); d1_clean, d2_clean = drug1.strip(), drug2.strip()
            drugs_in_cell_line.add(d1_clean); drugs_in_cell_line.add(d2_clean)
            combo_to_samples_map[frozenset([d1_clean, d2_clean])].append(sample)
        except (ValueError, AttributeError): continue
    all_drugs_list = sorted(list(drugs_in_cell_line)); random.shuffle(all_drugs_list)
    train_set, validation_set, test_set = [], [], []
    if experiment_type == 'new_combo':
        unique_combos = list(combo_to_samples_map.keys()); random.shuffle(unique_combos)
        n_total = len(unique_combos); train_end_idx = int(n_total * 0.6); val_end_idx = int(n_total * 0.8)
        train_keys = unique_combos[:train_end_idx]; val_keys = unique_combos[train_end_idx:val_end_idx]; test_keys = unique_combos[val_end_idx:]
        def get_samples_from_keys(keys, combo_map): return [s for k in keys for s in combo_map[k]]
        train_set = get_samples_from_keys(train_keys, combo_to_samples_map); validation_set = get_samples_from_keys(val_keys, combo_to_samples_map); test_set = get_samples_from_keys(test_keys, combo_to_samples_map)
        def get_original_combo_names_from_keys(keys, combo_map):
            original_names = set()
            for k_set in keys:
                if k_set in combo_map:
                    for sample in combo_map[k_set]:
                        if 'combo_name' in sample:
                            original_names.add(sample['combo_name'])
            return sorted(list(original_names))
        train_combo_names_original = get_original_combo_names_from_keys(train_keys, combo_to_samples_map)
        val_combo_names_original = get_original_combo_names_from_keys(val_keys, combo_to_samples_map)
        test_combo_names_original = get_original_combo_names_from_keys(test_keys, combo_to_samples_map)

        key_for_current_cell = {
            'train': train_combo_names_original,
            'val': val_combo_names_original,
            'test': test_combo_names_original
        }
    
    elif experiment_type == 'new_drug':
        if len(all_drugs_list) < 20: return [], [], []
        test10_pool = set(all_drugs_list[-10:]); v10_pool = set(all_drugs_list[-20:-10]); train_remainder_pool = set(all_drugs_list[:-20])
        train_combo_keys_set = []; val_combo_keys_set = []; test_combo_keys_set = []        
        def split_points(sample):
            coords = sample['sparse_coords_log']; responses = sample['sparse_y']
            min_c0 = np.min(coords[:, 0]); min_c1 = np.min(coords[:, 1]); mono_mask = (coords[:, 0] == min_c0) | (coords[:, 1] == min_c1)
            combo_mask = ~mono_mask; mono_data, combo_data = None, None
            if np.any(mono_mask):
                mono_data = sample.copy(); mono_data['sparse_coords_log'] = coords[mono_mask]; mono_data['sparse_y'] = responses[mono_mask]
            if np.any(combo_mask):
                combo_data = sample.copy(); combo_data['sparse_coords_log'] = coords[combo_mask]; combo_data['sparse_y'] = responses[combo_mask]
            return mono_data, combo_data
        for combo_key, samples in combo_to_samples_map.items():
            d1, d2 = tuple(combo_key)
            d1_is_val = d1 in v10_pool; d2_is_val = d2 in v10_pool; d1_is_test = d1 in test10_pool; d2_is_test = d2 in test10_pool
            for sample in samples:
                mono_data, combo_data = split_points(sample)
                if mono_data: train_set.append(mono_data)
                if combo_data:
                    is_test_related = d1_is_test or d2_is_test; is_val_related = d1_is_val or d2_is_val
                    if (d1_is_val and d2_is_test) or (d1_is_test and d2_is_val): 
                        validation_set.append(combo_data); test_set.append(combo_data)
                        val_combo_keys_set.append(combo_key); test_combo_keys_set.append(combo_key)
                    elif is_test_related: test_set.append(combo_data); test_combo_keys_set.append(combo_key)
                    elif is_val_related: validation_set.append(combo_data); val_combo_keys_set.append(combo_key)
                    else: train_set.append(combo_data); train_combo_keys_set.append(combo_key)       
        def get_original_combo_names_from_keys(keys, combo_map):
            original_names = set()
            for k_set in keys:
                if k_set in combo_map:
                    for sample in combo_map[k_set]:
                        if 'combo_name' in sample:
                            original_names.add(sample['combo_name'])
            return sorted(list(original_names))
        train_combo_names_original = get_original_combo_names_from_keys(train_combo_keys_set, combo_to_samples_map)
        val_combo_names_original = get_original_combo_names_from_keys(val_combo_keys_set, combo_to_samples_map)
        test_combo_names_original = get_original_combo_names_from_keys(test_combo_keys_set, combo_to_samples_map)
        
        key_for_current_cell = {
                        "train": train_combo_names_original,
                        "val": val_combo_names_original,
                        "test": test_combo_names_original
                                }
    
    train_points = sum(len(s['sparse_y']) for s in train_set); val_points = sum(len(s['sparse_y']) for s in validation_set); test_points = sum(len(s['sparse_y']) for s in test_set)
    print("[INFO] Data partitioning...Done.")
    print(f"- Training set: {train_points}, \n- Validation set: {val_points}, \n- Test set: {test_points}")
    
    return train_set, validation_set, test_set, key_for_current_cell