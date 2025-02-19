import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, filename, logger):
        """
        Initialization data file path and other data-related configurations 
        Read data from data file
        Preprocess the data
        """
        pass
    def __len__(self):
        """
        Dataset length
        """
        raise NotImplementedError
    def __getitem__(self, index):
        """
        Return a set of data pairs (data[index], label[index])
        """
        raise NotImplementedError
    
    @staticmethod 
    def collate_fn(batch_data):
        """
        As parameters to torch.utils.data.DataLoader, Preprocess batch_data
        """
        pass
    def __read_data(self):
        pass
    def __preprocess_data(self):
        pass 


class TravelDataset(BaseDataset):
    def __init__(self, encoder_data, decoder_data, lengths, time_data, dist1_data, dist2_data, z_data, z_time_data, z_dist1_data, z_dist2_data, trend_feature, trend_label):
        self.encoder_data = encoder_data
        self.decoder_data = decoder_data
        self.lengths = lengths
        self.time_data = time_data
        self.dist1_data = dist1_data
        self.dist2_data = dist2_data
        self.z_data = z_data
        self.z_time_data = z_time_data
        self.z_dist1_data = z_dist1_data
        self.z_dist2_data = z_dist2_data
        self.trend_feature = trend_feature
        self.trend_label = trend_label
    
    def __len__(self):
        return len(self.encoder_data)
    
    def __getitem__(self, idx):
        encoder_input = torch.tensor(self.encoder_data[idx], dtype=torch.long)   
        decoder_input = torch.tensor(self.decoder_data[idx], dtype=torch.long)
        length = torch.tensor(self.lengths[idx], dtype=torch.long)
        time_input = torch.tensor(self.time_data[idx], dtype=torch.long)        
        dist1_input = torch.tensor(self.dist1_data[idx], dtype=torch.float)
        dist2_input = torch.tensor(self.dist2_data[idx], dtype=torch.float)
        z_input = torch.tensor(self.z_data[idx], dtype=torch.long)
        z_time = torch.tensor(self.z_time_data[idx], dtype=torch.long)
        z_dist1 = torch.tensor(self.z_dist1_data[idx], dtype=torch.float)
        z_dist2 = torch.tensor(self.z_dist2_data[idx], dtype=torch.float)
        trend_feature = torch.tensor(self.trend_feature[idx], dtype=torch.float)
        trend_label = torch.tensor(self.trend_label[idx], dtype=torch.long)
        
        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'length': length,
            'time_input': time_input,
            'dist1_input': dist1_input,
            'dist2_input': dist2_input,
            'z_input': z_input,
            'z_time': z_time,
            'z_dist1': z_dist1,
            'z_dist2': z_dist2,
            'trend_feature': trend_feature,
            'trend_label': trend_label
        }
        

