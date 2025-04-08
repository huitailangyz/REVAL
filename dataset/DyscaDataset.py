import json
import os
import re
import json
from tqdm import tqdm
import torch
from .BaseDataset import BaseDataset

class DyscaDataset(BaseDataset):
    def __init__(self, dataset_name='Dysca', type='clean', data_root='./data/', **kwargs):
        super().__init__(dataset_name,**kwargs)
        self.dataset_name = dataset_name
        self.type = type
        self.data_root = data_root
        with open(os.path.join(self.data_root, 'Dysca', self.type, 'input_file.json')) as file:
            self.data = json.load(file)
        self.num_samples = len(self.data)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.num_samples:
            sample = self.data[self.index]
            sample['images'][0] = os.path.join(self.data_root, 'Dysca', self.type, sample['images'][0])
            self.index += 1
            return sample
        else:
            raise StopIteration
        
    def get_most_similar(self,prediction, choices):
        """
        Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction.
        """
        # !pip install python-Levenshtein
        from Levenshtein import distance
        distances = [distance(prediction, choice) for choice in choices]
        min_dis = min(distances)
        if distances.count(min_dis) > 1:
            ind  = len(distances) - distances[::-1].index(min_dis) - 1
        else:
            ind = distances.index(min(distances))
        return choices[ind]

    def normalize_extracted_answer(self, question, extraction, question_type) -> str:
        """
        Normalize the extracted answer to match the answer type
        """
        if question_type == 'multi choice' or 'true or false':
            # make sure the extraction is a string
            if isinstance(extraction, str):
                extraction = extraction.strip()
            else:
                try:
                    extraction = str(extraction)
                except:
                    extraction = ""
        
            # extract "A" from "(A) text"
            letter = re.findall(r'\(([a-zA-Z])\)', extraction)
            if len(letter) == 0:
                letter = re.findall(r'[A-Z]', extraction)
            if len(letter) > 0:
                extraction = letter[-1].upper()
             
            question = question.split("Choices:\n")[-1].split("\nHint:")[0]
            choices = question.split("\n")
            
            options = [chr(ord('A') + i) for i in range(len(choices))]
                
            if extraction in options:
                # convert option letter to text, e.g. "A" -> "text"
                ind = options.index(extraction)
                extraction = choices[ind].split(" ")[1:]
                extraction = ' '.join(extraction)
            else:
                # select the most similar option
                extraction = self.get_most_similar(extraction, choices)
                extraction = extraction.split(" ")[1:]
                extraction = ' '.join(extraction)

        return extraction
    
    def caption_sent_transformer(self,sent1,sent2):
        sent1 = [sent1]
        sent2 = [sent2]
        embed1 = self.model.encode(sent1,device='cuda:0')
        embed2 = self.model.encode(sent2,device='cuda:0')
        similarities = self.model.similarity(embed1, embed2)
        similarities = round(similarities.item(),2)
        return similarities

    def caption_clip(self,prediction,gt):
        prediction = prediction.split()
        if len(prediction)>30:
            prediction = prediction[:30]
        prediction = (" ").join(prediction)
        prediction_feature = self.processor(text = [prediction], return_tensors="pt", padding=True).to("cuda:0")
        prediction_embbed = self.model.get_text_features(**prediction_feature)
        gt_feature = self.processor(text = [gt], return_tensors="pt", padding=True).to("cuda:0")
        gt_embbed = self.model.get_text_features(**gt_feature)
        sim = torch.cosine_similarity(prediction_embbed,gt_embbed)

        return sim.cpu().item()
    
    def safe_equal(self, prediction, gt):
        """
        Check if the prediction is equal to the gt, even if they are of different types
        """
        try:
            if prediction == gt:
                return True
            return False
        except Exception as e:
            print(e)
        return False
        
    def calculate_result(self, results):
        """
        results: (json) the output of the dataset stored in json format, with each sample in the format {"id": XXX, "instruction": XXX, "in_images": XXX, "answer": XXX, "out_image": XXX, ... }
        Return: (dict) all required quantization results in dictionary format
        """
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2",device='cuda:0')

        question_majorities = ['movie', 'action', 'tv show', 'profession', 'landmark', 'anime', 
            'clothes', 'celebrity', 'food', 'plant', 'age', 'gender', 'emotion',
            'race', 'animal', 'object', 'style','text','background', 'color']
        res = {'name':'Model'}
        avg_mc,avg_tf,avg_ic=0.0,0.0,0.0
        nan_mc,nan_tf,nan_ic = 0,0,0
        for qm in tqdm(question_majorities):
            accuracy, mc, tf, ic = self.calculate_result_majority(question_majority=qm)
            res.update(accuracy)
            avg_mc += mc
            avg_tf += tf
            avg_ic += ic
            if ic == 0:
                nan_ic += 1
            if mc == 0:
                nan_mc += 1
            if tf == 0:
                nan_tf += 1

        avg_mc = round(avg_mc/(len(question_majorities)-nan_mc),2)
        avg_tf = round(avg_tf/(len(question_majorities)-nan_tf),2)
        avg_ic = round(avg_ic/(len(question_majorities)-nan_ic),2)
        avg_all = round((avg_mc+avg_tf+avg_ic)/3,3) # 这个是场景所有问题的平均准确率，如果仅要它，下面代码都可以注释掉
        res.update({'avg_mc':avg_mc})
        res.update({'avg_tf':avg_tf})
        res.update({'avg_ic':avg_ic})
        res.update({'avg_all':avg_all})
        return res

    def calculate_result_majority(self, question_majority=None):
        '''
        question_majority: (str) fine-grained level of recognition tasks, e.g., age, gender. Set None to compute on all questions.
        '''
        data = self.data

        total = 0
        multi_choice_correct,true_or_false_correct,caption_score = 0.0,0.0,0.0
        multi_choice_total,true_or_false_total,caption_total = 0.0,0.0,0

        for idx in range(len(self.results)):
            id_h = self.results[idx]['id']
            if data[id_h]["question_majority"] == question_majority:
                question_type = data[id_h]["question_type"]
                question = data[id_h]["instruction"]
                answer = self.results[idx]["answer"]
                gt = data[id_h]["answer"]
                if 'None' in answer:
                    continue
                total += 1
                if question_type == 'multi choice':
                    # normalize the extracted answer to match the answer type
                    prediction = self.normalize_extracted_answer(question,answer,question_type)

                    # verify the prediction is true or false
                    true_false = self.safe_equal(prediction, gt)
                    if answer == '':
                        true_false = False
                    if true_false:
                        multi_choice_correct+=1
                    multi_choice_total += 1
                elif question_type == "true or false":
                    # normalize the extracted answer to match the answer type
                    prediction = self.normalize_extracted_answer(question,answer,question_type)

                    # verify the prediction is true or false
                    true_false = self.safe_equal(prediction, gt)
                    if answer == '':
                        true_false = False
                    if true_false:
                        true_or_false_correct+=1
                    true_or_false_total += 1
                elif question_type == 'image caption':
                    caption_score += self.caption_sent_transformer(answer,gt)
                    caption_total += 1
                else:
                    pass
        if multi_choice_total == 0:
            multi_choice_accuracy = 0.0
        else:
            multi_choice_accuracy = round(multi_choice_correct / multi_choice_total * 100, 2)
        if true_or_false_total == 0:
            true_or_false_accuracy = 0.0
        else: 
            true_or_false_accuracy = round(true_or_false_correct / true_or_false_total * 100, 2)
        if caption_total == 0:
            caption_score_final = 0.0
        else: 
            caption_score_final = round(caption_score / caption_total * 100, 2)
        average = multi_choice_accuracy + true_or_false_accuracy + caption_score_final
        if question_majority == 'all':
            res = {
                'question_majority':question_majority, # fine-grained level
                'vqa_num':total, # the number of questions
                'multi_choice_accuracy':multi_choice_accuracy,
                'true_or_false_accuracy':true_or_false_accuracy,
                'caption_score':caption_score_final,
                'average':round(average/3,2)
            }
        else:
            res = {
                f'{question_majority}_multi_choice_accuracy':multi_choice_accuracy,
                f'{question_majority}_true_or_false_accuracy':true_or_false_accuracy,
                f'{question_majority}_caption_score':caption_score_final,
            }

        return res, multi_choice_accuracy, true_or_false_accuracy, caption_score_final