import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from geopy.distance import geodesic
from model.base_model import BaseModel
from model.embeddings import POIembedding, TimeEmbedding, SpaceEmbedding
from model.AE_model import MambaDecoder
from utils.util import sequence_mask
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

TINY = 1e-6

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class Mamba_Block(nn.Module):
    def __init__(self, d_model, d_state, 
                 expand, conv, depth=5):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'mamba': Mamba(d_model, d_state, expand, conv), 
                'norm2': nn.LayerNorm(d_model),  
                'mlp': nn.Sequential(          
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, d_model),
                    nn.Dropout(0.1)            
                )
            }))
        '''
        self.block = nn.ModuleList(
            [Mamba(d_model,
                   d_state,
                   expand,
                   conv) 
                   for _ in range(depth)]
        )
        '''

    def forward(self, x):
        for blk in self.blocks:
            norm1 = blk['norm1']
            mamba = blk['mamba']
            norm2 = blk['norm2']
            mlp = blk['mlp']
            x = mamba(norm1(x)) + x
            x = mlp(norm2(x)) + x
            #x = blk(x) + x
        return x
    
# replace for encoder and generator
class BiMamba(nn.Module):
    def __init__(self, d_model, d_intermediate, vocab_size, expand, conv_dim, tem_depth, p_dropout, d_trend=None):
        super().__init__()
        self.embeddings = POIembedding(vocab_size, d_model)  # POI embeddings
        self.time_embeddings = TimeEmbedding(24, 32)
        self.distance_embeddings = SpaceEmbedding(32)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 3*32, d_model + 3*32),
            nn.LayerNorm(d_model + 3*32),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout)
            )
        if d_trend is not None:
            self.fusionTrend = nn.Linear(d_model + 3*32 + d_trend, d_model + 3*32)
        self.ssm_forw = Mamba_Block(
            d_model + 3*32,
            d_intermediate,
            expand,
            conv_dim,
            tem_depth
            )
        self.ssm_back = Mamba_Block(
            d_model + 3*32,
            d_intermediate,
            expand,
            conv_dim,
            tem_depth
            )

    def forward(self, x, context, condition=None):
        '''
        Input: x: [batch_size,length]
               context: [3, batch_size, length]
               gamma: [list 2] for past feature [B T]
        Return: y: [batch_size, d_model+32+2*32]
        '''
        # Forward
        tensor = self.embeddings(x)  # shape: [batch_size, length, embedding_size]
        time_t = self.time_embeddings(context[0])  # context[0] -> time indices, shape: [batch_size, length, time_embedding_size]
        space = self.distance_embeddings(context[1], context[2])    #shape: [batch_size, length, space_embedding_size]
        tensor = torch.cat([tensor, time_t], dim=-1)  # Concatenate along the last dimension
        input_forw = torch.cat([tensor, space], dim=-1)

        #Backward
        _x_flip = torch.flip(x, dims=[1])
        _time_flip = torch.flip(context[0], dims=[1])
        _space1_flip = torch.flip(context[1], dims=[1])
        _space2_flip = torch.flip(context[2], dims=[1])
        _tensor_flip = self.embeddings(_x_flip)
        _time_t_flip = self.time_embeddings(_time_flip)
        _space_flip = self.distance_embeddings(_space1_flip, _space2_flip)
        _tensor_flip = torch.cat([_tensor_flip, _time_t_flip], dim=-1)
        input_back = torch.cat([_tensor_flip, _space_flip], dim=-1)

        # cooprate condition information
        if condition is not None:
            length = input_forw.shape[1]
            condition = condition[:, None, :].repeat(1, length, 1)
            concat_f = torch.cat((input_forw, condition), dim=-1)
            concat_b = torch.cat((input_back, condition), dim=-1)
            input_forw = self.fusionTrend(concat_f)
            input_back = self.fusionTrend(concat_b)

        # Mamba block
        x_ssm_forw = self.ssm_forw(input_forw)
        x_ssm_back = self.ssm_back(input_back)

        y = (x_ssm_forw + torch.flip(x_ssm_back, dims=[1])) # [batch_size, length, d_model+32+2*32]
        output_y = y.mean(dim=1)
        #output_y = y[:, -1, :]
        return output_y


class BiMambaAEModel(BaseModel):
    def __init__(self, max_distance, poi_coordinates, d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        expand: int, 
        conv_dim: int,
        tem_depth: int,
        p_dropout: float,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False, 
        device=None, 
        dtype=None):
        super().__init__()
        self.encoder = BiMamba(d_model, d_intermediate, vocab_size, expand, conv_dim, tem_depth, p_dropout)
        self.decoder = MambaDecoder(d_model=d_model, n_layer=n_layer, d_intermediate=d_intermediate, vocab_size=vocab_size, ssm_cfg=ssm_cfg, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg, norm_epsilon=norm_epsilon, rms_norm=rms_norm, initializer_cfg=initializer_cfg, fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32, device=device)
        self.poi_coordinates = poi_coordinates
        self.max_distance = max_distance
        self.poi_count_embedding = nn.Linear(1, 32)
        self.condition_projection = nn.Linear(d_model + 32 + 2*32 + 1, d_model + 32 + 2*32)
        self.device = device
    
    def calculate_distance(self, poi_id1, poi_id2):
        #Calculate haversine distance between two POIs
        lon1, lat1 = map(float, self.poi_coordinates[poi_id1])
        lon2, lat2 = map(float, self.poi_coordinates[poi_id2])
        coord1 = (lat1, lon1)
        coord2 = (lat2, lon2)
        distance = geodesic(coord1, coord2).kilometers
        return distance

    '''
    def normalize_distances(self, distances):
        min_dist = distances.min()
        max_dist = distances.max()
        return (distances - min_dist) / (max_dist - min_dist)
    '''
    def normalize_distances(self, distances):
        return 0.5 * distances / self.max_distance
    
    def get_disScore(self, end_poi_ids, batch_size):
        distances = torch.zeros(batch_size, self.decoder.lm_head.out_features, device=self.device)
        for i in range(batch_size):
            for poi in self.poi_coordinates:
                dis = self.calculate_distance(poi, str(end_poi_ids[i].item()))
                distances[i, int(poi)] = dis
        # normlization 
        for i in range(batch_size):
            min_dis = torch.min(distances[i]) 
            max_dis = torch.max(distances[i])
        
            norm_dis = (distances[i] - min_dis) / (max_dis - min_dis)
            distances[i] = norm_dis
        disScores = torch.exp((-0.5) / (TINY + distances ** 2))
        return disScores


    def forward(self, X, context, target_sequence_length, max_target_sequence_length, batch_size, go_int, pad_index):
        batch_size = X.shape[0]
        latent_vector = self.encoder(X, context)  # Get latent vector from encoder
        input = torch.full((batch_size, 1), go_int, device=self.device)  # shape: [batch_size, 1]
        #input = X[:, 0]
        outputs = torch.zeros(max_target_sequence_length, batch_size, self.decoder.lm_head.out_features, device=self.device)
        generated_tokens = torch.full((batch_size, max_target_sequence_length), pad_index, dtype=torch.int64, device=self.device)
        # Initializing the current distance to destination
        end_poi_ids = X[:, -1]
        start_poi_ids = X[:, 0] 
        current_distances1 = torch.zeros(batch_size, 1, device=self.device)
        current_distances2 = torch.zeros(batch_size, 1, device=self.device)
        # mask, 0 means invisible
        mask = torch.ones(batch_size, self.decoder.lm_head.out_features, device=self.device)
        mask[torch.arange(batch_size), end_poi_ids] = 0
        mask[torch.arange(batch_size), start_poi_ids] = 0

        # Sequentially generate each token
        '''
        i = 1
        generated_tokens[:, 0] = input
        outputs[0] = torch.eye(self.decoder.lm_head.out_features, device=self.device)[input.squeeze()].to(torch.float32)
        input = input.unsqueeze(1)
        if i < max_target_sequence_length:
        '''
        for i in range(max_target_sequence_length):
            # calculate poi_num and distance as condition
            poi_count = torch.full((batch_size, 1), max_target_sequence_length - i , device=self.device)  # shape: [batch_size, 1]
            poi_count_embedded = self.poi_count_embedding(poi_count.float())    # shape: [batch_size, 32]
            
            # Update current distances based on the last generated token
            if i > 0:
                current_poi_ids = generated_tokens[:, i - 1]  # Last generated POI ID
                for j in range(batch_size):
                    if str(current_poi_ids[j].item()) in self.poi_coordinates and str(end_poi_ids[j].item()) in self.poi_coordinates:
                        s_dis = self.calculate_distance(str(current_poi_ids[j].item()), str(start_poi_ids[j].item()))
                        e_dis = self.calculate_distance(str(current_poi_ids[j].item()), str(end_poi_ids[j].item()))
                        current_distances1[j] = self.normalize_distances(s_dis)
                        current_distances2[j] = self.normalize_distances(e_dis)
                        
            else:
                current_poi_ids = X[:, 0]
            
            #current_distances = self.normalize_distances(current_distances)
            #condition = torch.cat([poi_count_embedded, current_distances], dim=1)  # shape: [batch_size, d_model + 1]
            #condition = condition.float()
            condition = (poi_count_embedded, current_distances1, current_distances2)
            #latent_vector = torch.cat([latent_vector, condition], dim=1)  # shape: [batch_size, d_model + d_model + 1]
            #latent_vector = self.condition_projection(latent_vector)

            #dis_score = self.get_disScore(end_poi_ids, batch_size)

            output_logits, predict = self.decoder(input, latent_vector, condition)
            output_logits = output_logits.transpose(0,1) #shape: [length, batch_size, vocab_size]
            predict = predict.transpose(0,1)    #shape: [length, batch_size]
            outputs[i] = output_logits[-1]  # Only store the last generated logits

            # Get the predicted token and update the input for the next step
            next_token = predict[-1]
            generated_tokens[:, i] = next_token

            # Update input with the new token embedding for the next timestep
            input = torch.cat([input, next_token.unsqueeze(1)], dim=1)
            #i += 1
        outputs = outputs.transpose(0,1)
        
        for j in range(len(target_sequence_length)):
            leng = target_sequence_length[j]
            if (leng < max_target_sequence_length):
                generated_tokens[j, leng:] = pad_index
        return outputs, generated_tokens, latent_vector