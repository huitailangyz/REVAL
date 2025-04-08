import json
import os
import re

import numpy as np
from tqdm import tqdm
import time
from .BaseDataset import BaseDataset

"""
dataset format:
{
    "instruction": "How many people are in the picture?",
    "images": ["./dataset/test_images/2000x1372_wmkn_0012149409555.jpg",]
},
"""


def extract_answer(text):
    letter = re.findall(r'\(([a-zA-Z])\)', text)
    if len(letter) == 0:
        letter = re.findall(r'[A-Z]', text)
    if len(letter) > 0:
        pred = letter[-1].upper()
    else:
        pred = 'Z'

    return pred

class M3oralBenchDataset(BaseDataset):
    """text only"""
    def __init__(self, dataset_name='M3oralBench', data_root='./data/', type=None, **kwargs):
        super().__init__(dataset_name, **kwargs)
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.type = type
        if type is None:
            with open(os.path.join(self.data_root, 'M3oralBench/query.json'), 'r') as file:
                self.data = json.load(file)
        else:
            with open(os.path.join(self.data_root, 'M3oralBench/query_{}.json'.format(type)), 'r') as file:
                self.data = json.load(file)
        self.num_samples = len(self.data)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.num_samples:
            sample = self.data[self.index]
            sample['images'][0] = os.path.join(self.data_root, sample['images'][0])
            self.index += 1
            return sample
        else:
            raise StopIteration

    def calculate_result(self, results, **kwargs):
        """
        results: (json) the output of the dataset stored in json format, with each sample in the format of {"id": XXX, "instruction": XXX, "in_images": XXX, "answer": XXX, "out_image": XXX, ... }
        Return: (dict) all required quantization results in dictionary format
        """
        res = {'overall': 0.0}

        for i, result in tqdm(enumerate(results)):
            task_type = self.data[i]['type']
            foundation = self.data[i]['foundation']
            gt_choice = self.data[i]['gt_choice']
            text = result['answer']

            pred = extract_answer(text)

            if task_type not in res:
                res[task_type] = {}

            if foundation not in res[task_type]:
                res[task_type][foundation] = {'correct': 0, 'total': 0}

            if pred == gt_choice:
                res[task_type][foundation]['correct'] += 1

            res[task_type][foundation]['total'] += 1

        accs = []
        for task_type in res:
            for foundation in res[task_type]:
                correct = res[task_type][foundation]['correct']
                total = res[task_type][foundation]['total']
                accuracy = correct / total if total > 0 else 0
                res[task_type][foundation] = accuracy
                accs.append(accuracy)
        res['overall'] = sum(accs) / len(accs)

        return res
