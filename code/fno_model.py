import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models.fno import FNO

class FNOPointwiseModel(nn.Module):
    """
    Drug combination response prediction model based on FNO
    Input: Drug characteristics, logarithmic concentration coordinates, concentration boundaries
    Output: Predicted growth response values of drug combinations
    """
    def __init__(self, config):
        super().__init__()
        model_config = config['model']
        self.initial_grid_res = model_config['grid']
        self.fno_width = model_config['fno_width']

        input_dim = model_config["feature_dim"] * 2
        final_out_dim = self.fno_width * self.initial_grid_res * self.initial_grid_res
        lifting_layers = []
        lifting_hidden_dims = model_config['lifting']

        if isinstance(lifting_hidden_dims, int):
            lifting_hidden_dims = [lifting_hidden_dims]
        dropout_rate = model_config.get("dropout_rate1", 0.0)    

        for h_dim in lifting_hidden_dims:
            lifting_layers.append(nn.Linear(input_dim, h_dim))
            if dropout_rate > 0: lifting_layers.append(nn.Dropout(dropout_rate))
            lifting_layers.append(nn.GELU())
            input_dim = h_dim
        lifting_layers.append(nn.Linear(input_dim, final_out_dim))
        self.fno_lifting = nn.Sequential(*lifting_layers)
        self.fno_layers = FNO(
        n_modes=(model_config['fno_modes'], model_config['fno_modes']), 
        hidden_channels=model_config['fno_width'], in_channels=self.fno_width, out_channels=self.fno_width, 
        n_layers=model_config.get('fno_n_layers', 4),lifting=nn.Identity(),
        projection=nn.Identity())
        decoder_layers = []
        input_dim = self.fno_width
        dropout_rate = model_config.get("dropout_rate2", 0.0)
        for hidden_dim in model_config['decoder_hidden_dims']:
            decoder_layers.append(nn.Linear(input_dim, hidden_dim))
            decoder_layers.append(nn.GELU()) 
            if dropout_rate > 0: decoder_layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        decoder_layers.append(nn.Linear(input_dim, 1)); self.point_decoder = nn.Sequential(*decoder_layers)
        self._initialize_weights()
    def _initialize_weights(self):
            target_modules = [self.fno_lifting, self.point_decoder]
            for module in target_modules:
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                        if m.bias is not None: nn.init.constant_(m.bias, 0.0)
    def normalize_coords(self, log_coords, bounds):
        min_vals = bounds[:, :, 0].unsqueeze(1); max_vals = bounds[:, :, 1].unsqueeze(1)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0 
        return -1 + 2 * (log_coords - min_vals) / range_vals
    def forward(self, drug_features, sparse_coords_log, log_coord_bounds):
        B = drug_features.shape[0]
        if drug_features.dim() == 4: drug_features = drug_features.squeeze(1)
        drug_features_flat = drug_features.permute(0, 2, 1).reshape(B, -1)
        x = self.fno_lifting(drug_features_flat)
        x = x.view(B, self.fno_width, self.initial_grid_res, self.initial_grid_res)
        latent_map = self.fno_layers(x)
        normalized_coords = self.normalize_coords(sparse_coords_log, log_coord_bounds).unsqueeze(1)
        sampled_features = F.grid_sample(latent_map, normalized_coords, mode='bilinear', align_corners=False, padding_mode='border')
        sampled_features = sampled_features.squeeze(2).permute(0, 2, 1).reshape(-1, self.fno_width)
        predictions = self.point_decoder(sampled_features)
        return predictions.view(B, -1)