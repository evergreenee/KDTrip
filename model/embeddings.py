import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel

class Embeddings(BaseModel):
    def __init__(self, voc_poi_size, embedding_size, time_embedding_size=32, distance_embedding_size=32):
        super(Embeddings, self).__init__()
        self.poi_embeddings = nn.Embedding(voc_poi_size, embedding_size)
        self.time_embeddings = nn.Embedding(24, time_embedding_size)
        self.distance_embeddings1 = nn.Parameter(torch.randn(distance_embedding_size))
        self.distance_embeddings2 = nn.Parameter(torch.randn(distance_embedding_size))
        self.linear = nn.Linear(embedding_size, embedding_size)

    def forward(self, X, time_context, distance_context):
        poi_embed = self.poi_embeddings(X)
        time_embed = self.time_embeddings(time_context)
        space_embed = torch.tensordot(distance_context[0], self.distance_embeddings1, dims=0) + torch.tensordot(distance_context[1], self.distance_embeddings2, dims=0)
        #combined_embed = torch.cat([poi_embed, time_embed, space_embed], dim=2)
        return self.linear(poi_embed), time_embed, space_embed
    
class POIembedding(BaseModel):
    def __init__(self, voc_poi_size, embedding_size):
        super(POIembedding, self).__init__()
        self.poi_embeddings = nn.Embedding(voc_poi_size, embedding_size)
        self.linear = nn.Linear(embedding_size, embedding_size)

    def forward(self, X):
        poi_embed = self.poi_embeddings(X)
        return self.linear(poi_embed)
    
    @property
    def weight(self):
        return self.poi_embeddings.weight

class TimeEmbedding(BaseModel):
    def __init__(self, time_num=24, time_embedding_size=32):
        super(TimeEmbedding, self).__init__()
        self.time_embeddings = nn.Embedding(time_num, time_embedding_size)

    def forward(self, time_context):
        time_embed = self.time_embeddings(time_context)
        return time_embed
    
class SpaceEmbedding(BaseModel):
    def __init__(self, distance_embedding_size=32):
        super(SpaceEmbedding, self).__init__()
        # Define embeddings as linear transformations for distance contexts
        self.distance_linear1 = nn.Linear(1, distance_embedding_size)  # For distance_context1
        self.distance_linear2 = nn.Linear(1, distance_embedding_size)  # For distance_context2

    def forward(self, distance_context1, distance_context2):
        # Ensure distance contexts have shape [batch_size, 1]
        distance_context1 = distance_context1.unsqueeze(-1) if distance_context1.dim() == 2 else distance_context1
        distance_context2 = distance_context2.unsqueeze(-1) if distance_context2.dim() == 2 else distance_context2

        # Transform distance contexts to embedding space
        distance_embed1 = self.distance_linear1(distance_context1)  # shape: [batch_size, sequence_length, embedding_size]
        distance_embed2 = self.distance_linear2(distance_context2)

        # Combine embeddings 
        space_embed = torch.cat([distance_embed1, distance_embed2], dim=-1)  # shape: [batch_size, sequence_length, embedding_size * 2]
        
        return space_embed
