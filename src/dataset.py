# 1.定义Dataset
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import JiebaTokenizer
import config

class ReviewAnalyDataset(Dataset):
    def __init__(self,path):
        self.data = pd.read_json(path,lines=True,orient="records").to_dict(orient="records")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]["review"],dtype = torch.long)
        target_tensor = torch.tensor(self.data[index]["label"],dtype = torch.float)
        return input_tensor, target_tensor


def get_dataloader(train = True):
    path = config.PROCESSED_DATA_DIR / ("train.jsonl" if train else "test.jsonl")
    dataset = ReviewAnalyDataset(path=path)
    return DataLoader(dataset,batch_size = config.BATCH_SIZE ,shuffle = True)

if __name__=='__main__':
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(False)
    print(len(train_dataloader))
    print(len(test_dataloader))
    for  input_tensor,target_tensor in train_dataloader:
        print(input_tensor.shape,target_tensor.shape)
        break