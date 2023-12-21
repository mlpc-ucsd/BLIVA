
import torch

from bliva.datasets.datasets.base_dataset import BasePromptDataset


import os
import json

from PIL import Image
import numpy as np

class OCRVQADataset(BasePromptDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.prompts =  ["{}","Question: {}", 
            "{} A short answer to the question is",
            "Q: {} A:",
            "Question: {} Short answer:",
            "Given the image, answer the following question with no more than three words. {}",
            "Based on the image, respond to this question with a short answer: {}. Answer:",
            "Use the provided image to answer the question: {} Provide your answer as short as possible:",
            "What is the answer to the following question?{}",
            "The question {} can be answered using the image. A short answer is"]
        self.ann_list = list(self.annotation.keys())
    def __len__(self):
        return len(self.annotation)

    def _add_instance_ids(self, key="instance_id"):
        # for idx, ann in enumerate(self.annotation['data']):
        #     ann[key] = str(idx)
        pass
            
    def __getitem__(self, index):
        image_key = self.ann_list[index]
        ann = self.annotation[image_key]

        image_path = os.path.join(self.vis_root, str(image_key) + '.jpg')
        try:
            image = Image.open(image_path).convert("RGB")
    
            image = self.vis_processor(image)
            question = self.text_processor(ann["questions"][0])
    
            choice = np.random.choice(len(self.prompts))
    
            text_input = self.prompts[choice].format(question)
    
            answer = ann["answers"][0]
            if len(ann["answers"]) > 1:
                answer_weight = {}
                for answer in ann["answers"]:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1 / len(ann["answers"])
                    else:
                        answer_weight[answer] = 1 / len(ann["answers"])
                
                answer = max(answer_weight, key=answer_weight.get)
        except Exception as e:
            return {
                
            }

        return {
            "image": image,
            "text_input": text_input,
            "text_output": answer,
        }

    def collater(self, samples):
        image_list, question_list, answer_list = [], [], [],

        for sample in samples:
            image_list.append(sample["image"])
           
            question_list.append(sample["text_input"])

            answers = sample["text_output"]

            answer_list.append(answers)
        

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
        }
        
class STVQADataset(BasePromptDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.prompts =  ["{}","Question: {}", 
            "{} A short answer to the question is",
            "Q: {} A:",
            "Question: {} Short answer:",
            "Given the image, answer the following question with no more than three words. {}",
            "Based on the image, respond to this question with a short answer: {}. Answer:",
            "Use the provided image to answer the question: {} Provide your answer as short as possible:",
            "What is the answer to the following question?{}",
            "The question {} can be answered using the image. A short answer is"]
    
    def __getitem__(self, index):
        ann = self.annotation['data'][index]

        image_path = os.path.join(self.vis_root, ann["file_path"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        choice = np.random.choice(len(self.prompts))

        text_input = self.prompts[choice].format(question)

        answer = ann["answers"][0]
        if len(ann["answers"]) > 1:
            answer_weight = {}
            for answer in ann["answers"]:
                if answer in answer_weight.keys():
                    answer_weight[answer] += 1 / len(ann["answers"])
                else:
                    answer_weight[answer] = 1 / len(ann["answers"])
            
            answer = max(answer_weight, key=answer_weight.get)

        return {
            "image": image,
            "text_input": text_input,
            "text_output": answer,
        }

    def collater(self, samples):
        image_list, question_list, answer_list = [], [], [],

        for sample in samples:
            image_list.append(sample["image"])
           
            question_list.append(sample["text_input"])

            answers = sample["text_output"]

            answer_list.append(answers)
        

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
        }
        
class DocVQADataset(BasePromptDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.prompts =  ["{}","Question: {}", 
            "{} A short answer to the question is",
            "Q: {} A:",
            "Question: {} Short answer:",
            "Given the image, answer the following question with no more than three words. {}",
            "Based on the image, respond to this question with a short answer: {}. Answer:",
            "Use the provided image to answer the question: {} Provide your answer as short as possible:",
            "What is the answer to the following question?{}",
            "The question {} can be answered using the image. A short answer is"]
    
    def __getitem__(self, index):
        ann = self.annotation['data'][index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        choice = np.random.choice(len(self.prompts))

        text_input = self.prompts[choice].format(question)

        answer = ann["answers"][0]
        if len(ann["answers"]) > 1:
            answer_weight = {}
            for answer in ann["answers"]:
                if answer in answer_weight.keys():
                    answer_weight[answer] += 1 / len(ann["answers"])
                else:
                    answer_weight[answer] = 1 / len(ann["answers"])
            
            answer = max(answer_weight, key=answer_weight.get).lower()

        return {
            "image": image,
            "text_input": text_input,
            "text_output": answer,
        }

    def collater(self, samples):
        image_list, question_list, answer_list = [], [], [],

        for sample in samples:
            image_list.append(sample["image"])
           
            question_list.append(sample["text_input"])

            answers = sample["text_output"]

            answer_list.append(answers)
        

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
        }