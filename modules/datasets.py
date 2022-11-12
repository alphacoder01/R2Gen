import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.args = args
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class MultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        if self.args.use_xrayvision:
            image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('L')
            image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('L')
        else:
            image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            try:
                image_1 = self.transform(image=np.array(image_1))["image"]
                image_2 = self.transform(image=np.array(image_2))["image"]
            except:
                image_1 = self.transform(image_1)
                image_2 = self.transform(image_2)
        if self.args.use_xrayvision:
            image = torch.stack((image_1, image_2), 0).float()
        else:
            image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class SingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            try:
                image = self.transform(image=np.array(image))["image"]
            except:
                image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
