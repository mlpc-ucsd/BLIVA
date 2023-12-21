
import torch

from bliva.datasets.datasets.base_dataset import BaseDataset


import os
import json

from PIL import Image
import numpy as np
import torch

class LLAVADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
    
    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, 'train2017/' + ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann['conversations'][0]["value"])

        answer = ann['conversations'][1]["value"]

        return {
            "image": image,
            "text_input": question,
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