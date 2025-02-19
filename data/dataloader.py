import torch
from torch.utils.data import DataLoader
from data.dataset import TravelDataset

class BaseDataLoader(object):
    """
    Nonuse torch.utils.data
    """
    def __init__(self, filename, batch_size, shuffle, logger):
        """
        Initialization data file path, batch data size, shuffle data
        Read data from data file
        Preprocess the data
        Spilt the data according to batch_size
        """
        pass
    def __len__(self):
        """
        How many batch
        """
        raise NotImplementedError
    def __getitem__(self, index):
        """
        Return batch_size data pairs
        """
        raise NotImplementedError
    def __read_data(self,):
        pass
    def __preprocess_data(self,):
        pass

def create_dataloaders(train_variables, test_variables, batch_size):
    encoder_train, decoder_train, train_lengths, train_time, train_dist1, train_dist2, z_train, z_train_time, z_train_dist1, z_train_dist2, trend_train_feature, trend_train_label = train_variables
    encoder_test, decoder_test, test_lengths, test_time, test_dist1, test_dist2, z_test, z_test_time, z_test_dist1, z_test_dist2, trend_test_feature, trend_test_label = test_variables
    
    train_dataset = TravelDataset(encoder_train, decoder_train, train_lengths, train_time, train_dist1, train_dist2, z_train, z_train_time, z_train_dist1, z_train_dist2, trend_train_feature, trend_train_label)
    test_dataset = TravelDataset(encoder_test, decoder_test, test_lengths, test_time, test_dist1, test_dist2, z_test, z_test_time, z_test_dist1, z_test_dist2, trend_test_feature, trend_test_label)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader