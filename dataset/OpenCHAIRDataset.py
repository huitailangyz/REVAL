import json
import os
import time

from tqdm import tqdm
from .BaseDataset import BaseDataset
from tqdm.auto import tqdm
from tqdm.contrib import tzip
"""
dataset format:
{
    "instruction": "How many people are in the picture?",
    "images": ["./dataset/test_images/2000x1372_wmkn_0012149409555.jpg",]
},
"""


def is_concrete(noun, concretness, t=4.5):
    if noun in concretness:
        return concretness[noun] > t
    return False


def extract_objs(captions, conc_df):
    import en_core_web_sm
    nlp = en_core_web_sm.load()
    objs = []
    for caption in tqdm(captions):
        doc = nlp(caption.lower())
        cur_objs = [token.lemma_ for token in doc if token.pos_ == 'NOUN' and is_concrete(token.lemma_, conc_df)]
        objs.append(cur_objs)
    return objs


def make_prompt(cap, obj):
    prompt = f"""[INST]
                 An image has the following caption: "{cap}"
                 Does the image contain the following object? "{obj}"
                 Answer yes/no/unsure.
                 [/INST]
                 The answer is: "
                 """.strip()
    return prompt


def get_answer(cap, obj):
    import openai
    prompt = make_prompt(cap, obj)
    openai.api_key = 'YOUR_OPENAI_KEY'
    response = None
    while response is None:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4o',
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
        except Exception as e:
            print(e)
            print('retrying...')
            time.sleep(10)
            continue
    response = response.choices[0].text[len(prompt):].strip()
    return response


def get_llm_responses(gt_captions, generated_objs):
    hits = []
    for cap, objs in tzip(gt_captions, generated_objs):
        objs = set(objs)
        for obj in objs:
            hits.append(get_answer(cap, obj))
    return hits


def get_och_score(responses):
    import pandas as pd
    data = pd.Series(responses).str.lower().str.strip()
    dv = data.value_counts()
    d = dv.to_dict()
    return 1 - d['yes'] / dv.sum()


class OpenCHAIRDataset(BaseDataset):
    def __init__(self, dataset_name='OpenCHAIR', data_root='./data/', **kwargs):
        super().__init__(dataset_name, **kwargs)
        self.dataset_name = dataset_name
        self.data_root = data_root
        with open(os.path.join(self.data_root, 'OpenCHAIR/query.json'), 'r') as file:
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
        import pandas as pd
        word_conc = pd.read_excel(os.path.join(self.data_root, 'OpenCHAIR/Concreteness_ratings_Brysbaert_et_al_BRM.xlsx'))[['Word', 'Conc.M']].set_index("Word").to_dict()[
            'Conc.M']
        generated_captions = []
        gt_captions = []
        for i, result in enumerate(results):
            gt_captions.append(self.data[i]['gt_caption'])
            generated_captions.append(result['answer'])

        generated_objs = extract_objs(generated_captions, word_conc)

        llm_responces = get_llm_responses(gt_captions, generated_objs)
        OpenCHAIR_score = get_och_score(llm_responces)

        res = {'OpenCHAIR_score': OpenCHAIR_score}

        return res
