import json
import time
from tqdm import tqdm
import os
from .BaseDataset import BaseDataset

"""
dataset format:
{
    "instruction": "How many people are in the picture?",
    "images": ["./dataset/test_images/2000x1372_wmkn_0012149409555.jpg",]
},
"""

template = """Given the input instruction, ground truth answer and detailed image information, please determine whether the response provided by a Large Vision-Language Model (LVLM) contains any hallucination. Hallucination here refers to the situation that the generated response is inconsistent with the input image. 

Please note that the ground truth answer and image information only contain factual information and may not be completely comprehensive in describing all the objects and their attributes. Detailed ana-lysis or reasoning in LVLM response should be encouraged and not considered as hallucination.

To evaluate the LVLM responses, you need to provide brief evidence to support your judgment. 

###Evaluation criteria:
-Without hallucination: The LVLM response is semantically similar to the ground truth answer and does not contain any contradictory factual claim with the provided information.
-With hallucination: The LVLM response is completely different from the ground truth answer, or contains a contradictory factual claim about an object, action, attribute, or any other detail that is not grounded in the provided information.

###Instruction:\n{}\n
###Ground Truth:\n{}\n
###Image Caption:\n{}\n
###Image Details:\n{}\n
###Model Response:\n{}
###Output Format:\n{}\n
"""

template4existence = """Given the input instruction and detailed image information, please determine whether the response provided by a Large Vision-Language Model (LVLM) contains any hallucination. Hallucination here refers to the situation that the generated response is inconsistent with the input image. 

Please note that the image information only contain factual information and may not be completely comprehensive in describing all the objects and their attributes. Detailed analysis or reasoning in LVLM response should be encouraged and not considered as hallucination.

To evaluate the LVLM responses, you need to provide brief evidence to support your judgment. 

###Evaluation criteria:
-Without hallucination: The LVLM response does not contain any contradictory factual claim with the provided information.
-With hallucination: The LVLM response contains a contradictory factual claim about an object, action, attribute, or any other detail that is not grounded in the provided information.

###Instruction:\n{}\n
###Image Caption:\n{}\n
###Image Details:\n{}\n
###Model Response:\n{}
###Output Format:\n{}\n
"""

class HQHDataset(BaseDataset):
    def __init__(self, dataset_name='HQH', data_root='./data/', **kwargs):
        super().__init__(dataset_name, **kwargs)
        self.dataset_name = dataset_name
        self.data_root = data_root
        with open(os.path.join(self.data_root, 'HQHxxl/query.json'), 'r') as file:
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
        import openai
        hal_results = {'overall': [], 'action': [], 'attribute': [], 'comparison': [], 'count': [], 'environment': [],
                       'existence': [], 'OCR': [], 'relation': []}
        openai.api_key = 'YOUR_OPENAI_KEY'
        hal_rates = {}

        with open(os.path.join(self.data_root, './image_data.json'), 'r', encoding='utf-8') as f:
            image_datas = json.load(f)
        for i, result in tqdm(enumerate(results)):
            image_id = self.data[i]['image_id']
            hal_type = self.data[i]['type']
            image_data = image_datas[str(image_id)]

            if hal_type == 'existence':
                prompt = template4existence.format(self.data[i]['instruction'], image_data['caption'],
                                                   image_data['image_information'], result['answer'])
            else:
                prompt = template.format(self.data[i]['instruction'], self.data[i]['ground_truth'], image_data['caption'],
                                         image_data['image_information'], result['answer'])

            gpt_output = None
            while gpt_output is None:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    gpt_output = response.choices[0].message.content

                except Exception as e:
                    print(e)
                    time.sleep(10)

            if 'With hallucination' in gpt_output:
                hal_results[hal_type].append(1)
                hal_results['overall'].append(1)
            elif 'Without hallucination' in gpt_output:
                hal_results[hal_type].append(0)
                hal_results['overall'].append(0)

        for hal_type in hal_results.keys():
            hal_rates[hal_type] = sum(hal_results[hal_type]) / len(hal_results[hal_type])

        return hal_rates
