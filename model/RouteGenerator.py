import torch
import torch.nn as nn
import torch.nn.functional as F
from geopy.distance import geodesic
from model.base_model import BaseModel
from utils.util import sequence_mask, get_delta

    
class RG_BiMamba(BaseModel):
    def __init__(self, vocab_size, d_model, max_distance, poi_coordinates, generator, decoder, feature_enc, trend_predict):
        super().__init__()

        self.encoder = generator
        self.decoder = decoder
        self.featureEnc = feature_enc
        self.trendPre = trend_predict
        self.fusionTrend = nn.Linear(d_model + 3*32 + self.featureEnc.hidden_dim, d_model + 3*32)
        self.poi_coordinates = poi_coordinates
        self.max_distance = max_distance
        self.poi_count_embedding = nn.Linear(1, 32)
        self.condition_projection = nn.Linear(d_model + 32 + 2*32 + 1, d_model + 32 + 2*32)
        self.vocab_size = vocab_size
    
    def calculate_distance(self, poi_id1, poi_id2):
        #Calculate haversine distance between two POIs
        lon1, lat1 = map(float, self.poi_coordinates[poi_id1])
        lon2, lat2 = map(float, self.poi_coordinates[poi_id2])
        coord1 = (lat1, lon1)
        coord2 = (lat2, lon2)
        distance = geodesic(coord1, coord2).kilometers
        return distance

    def normalize_distances(self, distances):
        return 0.5 * distances / self.max_distance
    
    def forward(self, X, context, trend_feature, target_sequence_length, max_target_sequence_length, batch_size, go_int, pad_index):
        batch_size = X.shape[0]
        outputs = torch.zeros(max_target_sequence_length, batch_size, self.decoder.lm_head.out_features, device=X.device)
        generated_tokens = torch.full((batch_size, max_target_sequence_length), pad_index, dtype=torch.int64, device=X.device)

        
        latent_vector = self.encoder(X, context)
        trend_vector = self.featureEnc(trend_feature)
        vec = torch.cat((latent_vector, trend_vector), dim=-1)
        latent_vector = self.fusionTrend(vec)
        #latent_vector = self.encoder(X, context, delta)  # Get latent vector from encoder
        input = torch.full((batch_size, 1), go_int, device=X.device)  # shape: [batch_size, 1]
        #input = X[:, 0]
        #outputs = torch.zeros(max_target_sequence_length, batch_size, self.decoder.lm_head.out_features, device=X.device)
        
        # Initializing the current distance to destination
        end_poi_ids = X[:, -1]
        start_poi_ids = X[:, 0] 
        current_distances1 = torch.zeros(batch_size, 1, device=X.device)
        current_distances2 = torch.zeros(batch_size, 1, device=X.device)

        # Sequentially generate each token
        
        '''
        i = 1
        generated_tokens[:, 0] = input
        outputs[0] = torch.eye(self.decoder.lm_head.out_features, device=X.device)[input.squeeze()].to(torch.float32)
        input = input.unsqueeze(1)
        if i < max_target_sequence_length:
        '''
        for i in range(max_target_sequence_length):
            # calculate poi_num and distance as condition
            poi_count = torch.full((batch_size, 1), max_target_sequence_length - i , device=X.device)  # shape: [batch_size, 1]
            poi_count_embedded = self.poi_count_embedding(poi_count.float())
            
            # Update current distances based on the last generated token
            if i > 0:
                current_poi_ids = generated_tokens[:, i - 1]  # Last generated POI ID
            else:
                current_poi_ids = X[:, 0]
            for j in range(batch_size):
                if current_poi_ids[j].item() in self.poi_coordinates and end_poi_ids[j].item() in self.poi_coordinates:
                    s_dis = self.calculate_distance(str(current_poi_ids[j].item()), str(start_poi_ids[j].item()))
                    e_dis = self.calculate_distance(str(current_poi_ids[j].item()), str(end_poi_ids[j].item()))
                    current_distances1[j] = self.normalize_distances(s_dis)
                    current_distances2[j] = self.normalize_distances(e_dis)
            #current_distances = self.normalize_distances(current_distances)
            #condition = torch.cat([poi_count_embedded, current_distances], dim=1)  # shape: [batch_size, d_model + 1]
            #condition = condition.float()
            condition = (poi_count_embedded, current_distances1, current_distances2)
            #latent_vector = torch.cat([latent_vector, condition], dim=1)  # shape: [batch_size, d_model + d_model + 1]
            #latent_vector = self.condition_projection(latent_vector)

            output_logits, predict = self.decoder(input, latent_vector, condition)
            output_logits = output_logits.transpose(0,1) #shape: [length, batch_size, vocab_size]
            predict = predict.transpose(0,1)    #shape: [length, batch_size]
            #print(output_logits[-1].shape)
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

        trend_predict = self.trendPre(trend_vector)
        return outputs, generated_tokens, trend_predict, latent_vector