import torch
import torch.nn as nn

class TrajFeatureEnc(nn.Module):
    def __init__(self, n_startPOI_ID, n_startPOI_Cat, n_endPOI_ID, n_endPOI_Cat, n_traj_len, embedding_dim=16, hidden_dim=512):
        super(TrajFeatureEnc, self).__init__()
        # embedding layer for traj feature
        self.startPOI_ID_embedding = nn.Embedding(n_startPOI_ID, embedding_dim)
        self.startPOI_Cat_embedding = nn.Embedding(n_startPOI_Cat, embedding_dim)
        self.endPOI_ID_embedding = nn.Embedding(n_endPOI_ID, embedding_dim)
        self.endPOI_Cat_embedding = nn.Embedding(n_endPOI_Cat, embedding_dim)
        self.traj_len_embedding = nn.Embedding(n_traj_len, embedding_dim)
        self.hidden_dim = hidden_dim
        # feature fusion
        self.fc = nn.Linear(5 * embedding_dim + 3, hidden_dim)  

    def forward(self, x):
        startPOI_time, endPOI_time, distance_to_end, startPOI_ID, startPOI_Cat, endPOI_ID, endPOI_Cat, traj_len= x.split(1, dim=1)
        
        startPOI_ID_embed = self.startPOI_ID_embedding(startPOI_ID.long()).squeeze(1)
        startPOI_Cat_embed = self.startPOI_Cat_embedding(startPOI_Cat.long()).squeeze(1)
        endPOI_ID_embed = self.endPOI_ID_embedding(endPOI_ID.long()).squeeze(1)
        endPOI_Cat_embed = self.endPOI_Cat_embedding(endPOI_Cat.long()).squeeze(1)
        traj_len_embed = self.traj_len_embedding(traj_len.long()).squeeze(1)
        
        # concat
        embedded = torch.cat((startPOI_ID_embed, startPOI_Cat_embed, endPOI_ID_embed, endPOI_Cat_embed, traj_len_embed,
                              startPOI_time, endPOI_time, distance_to_end), dim=1)
        trend_vec = self.fc(embedded)
        return trend_vec

class TrendPredict(nn.Module):
    def __init__(self, in_dim=512, out_dim=4, *args, **kwargs):
        super(TrendPredict, self).__init__(*args, **kwargs)
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, out_dim)  # 4 classes for trend: increasing, decreasing, increasing_then_decreasing, irregular

    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        x = self.fc3(x)
        
        return x