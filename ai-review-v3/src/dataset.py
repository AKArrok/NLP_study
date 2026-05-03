import json

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import config


class ReviewDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        return {
            'input_ids': torch.tensor(row['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(row['attention_mask'], dtype=torch.long),
            'label': torch.tensor(row['label'], dtype=torch.long),
        }


def get_dataset(train=True):
    path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    return ReviewDataset(path)


def get_dataloader(train=True):
    dataset = get_dataset(train=train)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    dataloader = get_dataloader(train=True)
    for batch in dataloader:
        print({k: v.shape for k, v in batch.items()})
        break
