import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, inputs):
        """
        Forward pass logic

        return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, dynamic_training=True):
        super(MyModel, self).__init__()
        self.dynamic_training = dynamic_training
        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)  # Assume keep_prob = 0.5
        if self.dynamic_training:
            # Embedding layers for POI, time, and distances
            self.poi_embedding = nn.Embedding(vocab_size, embedding_size)
            self.time_embedding = nn.Embedding(24, 32)
            self.distance_embedding1 = nn.Parameter(torch.randn(32))
            self.distance_embedding2 = nn.Parameter(torch.randn(32))
            
            # Weights and bias for transforming the POI embeddings(use linear replace for weights and bias)
            self.linear1 = nn.Linear(embedding_size, embedding_size)
            

        else:
            # Use pre-loaded embeddings from a file
            self.poi_embedding = nn.Embedding(vocab_size, embedding_size)  # You can still use nn.Embedding
            self.load_pretrained_embeddings('data/embedding_name_vec.dat')

    def load_pretrained_embeddings(self, file_path):
        """Load pretrained embeddings from a file."""
        # 如果要用预训练embedding，后续修改这里即可
        embeddings = []
        with open(file_path, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()[1:]))
                embeddings.append(values)
        pretrained_embeddings = torch.tensor(embeddings)
        self.poi_embedding.weight.data.copy_(pretrained_embeddings)

    def forward(self, poi_input, time_input):
        #修改，应该还要加distance输入等
        # input_X,input_X_de,input_t,input_d1,input_d2,target_sequence_length,z,z_t,z_d1,z_d2等
        if self.dynamic_training:
            # Apply embedding lookup for POI and time
            poi_embedded = self.poi_embedding(poi_input)
            poi_transformed = self.linear1(poi_embedded)

            time_embedded = self.time_embedding(time_input)

            return poi_transformed, time_embedded
        else:
            poi_embedded = self.poi_embedding(poi_input)
            return poi_embedded

