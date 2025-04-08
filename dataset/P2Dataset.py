import json
import os
from .BaseDataset import BaseDataset


class P2Dataset(BaseDataset):
    def __init__(self, dataset_name='p2', type='Q_Insensitive', data_root='./data/', **kwargs):
        super().__init__(dataset_name, **kwargs)
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.type = type
        self.data = []
        json_list = sorted(os.listdir(os.path.join(self.data_root, 'P2/json')))
        for j in json_list:
            json_path = os.path.join(self.data_root, 'P2/json', j)
            with open(json_path, 'r') as file:
                data_t = json.load(file)
            self.data.append(data_t)
        with open(os.path.join(self.data_root, 'P2/awareness_question.json'), 'r') as file:
            data_append = json.load(file)
        self.data += data_append
        self.num_samples = len(self.data)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.num_samples:
            data_dict = self.data[self.index]
            sample = {}
            sample["images"] = [os.path.join(self.data_root, 'P2', data_dict["images"][0])]
            sample["instruction"] = data_dict["instruction"].replace("[Q]", data_dict[self.type])
            self.index += 1
            return sample
        else:
            raise StopIteration


    def calculate_result(self, results, ans_cut_off=True, **kwargs):
        trues = 0
        if self.question_type == "Q_Insensitive":
            for res in results:
                if "no" in res["answer"].lower():
                    trues += 1
        else:
            for res in results:
                if "yes" in res["answer"].lower():
                    trues += 1
        res = {
            'accuracy': trues / self.num_samples,
        }
        return res

