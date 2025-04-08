import json
import os
from .BaseDataset import BaseDataset

"""
dataset format:
{
    "instruction": "How many people are in the picture?",
    "images": ["./dataset/test_images/2000x1372_wmkn_0012149409555.jpg",]
},
"""


class POPEDataset(BaseDataset):
    def __init__(self, dataset_name='POPE', type='adversarial', data_root='./data/', **kwargs):
        """
        type: 'adversarial', 'popular', 'random'
        """
        super().__init__(dataset_name, **kwargs)
        self.data_root = data_root
        with open(os.path.join(self.data_root, './POPE/coco_pope_{}.json'.format(type)), 'r') as file:
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
        pred_list = []
        label_list = []
        for result in results:
            text = result['answer']
            # Only keep the first sentence
            if text.find('.') != -1:
                text = text.split('.')[0]
            text = text.replace(',', '')
            words = text.split(' ')
            if 'No' in words or 'not' in words or 'no' in words:
                result['answer'] = 'no'
                pred_list.append(0)
            else:
                result['answer'] = 'yes'
                pred_list.append(1)

        for i in range(len(self.data)):
            if self.data[i]['label'] == 'no':
                label_list.append(0)
            else:
                label_list.append(1)
        pos = 1
        neg = 0
        yes_ratio = pred_list.count(1) / len(pred_list)

        TP, TN, FP, FN = 0, 0, 0, 0
        for pred, label in zip(pred_list, label_list):
            if pred == pos and label == pos:
                TP += 1
            elif pred == pos and label == neg:
                FP += 1
            elif pred == neg and label == neg:
                TN += 1
            elif pred == neg and label == pos:
                FN += 1

        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        acc = (TP + TN) / (TP + TN + FP + FN)
        res = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'yes_ratio': yes_ratio,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
        }
        return res
