"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from bliva.datasets.datasets.base_dataset import BaseDataset, BasePromptDataset
from PIL import Image
import numpy as np
import torch

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )

class TextCapsDataset(BasePromptDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
    
        self.prompts = [
            "A short image caption:",
            "A short image description:",
            "A photo of",
            "An image that shows",
            "Write a short description for the image.",
            "Write a description for the photo.",
            "Provide a description of what is presented in the photo.",
            "Briefly describe the content of the image.",
            "Can you briefly explain what you see in the image?",
            "Could you use a few words to describe what you perceive in the photo?",
            "Please provide a short depiction of the picture.",
            "Using language, provide a short account of the image.",
            "Use a few words to illustrate what is happening in the picture."
        ]

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation['data'][index]

        image_path = os.path.join(self.vis_root, ann["image_id"] + '.jpg')
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        text_output  = self.text_processor(ann["caption_str"])

        choice = np.random.choice(len(self.prompts))

        text_input = self.prompts[choice]

        return {
            "image": image,
            "text_input": text_input,
            #"image_id": self.img_ids[ann["image_id"]],
            'text_output': text_output,
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
        

class CaptionDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # self.img_ids = {}
        # n = 0
        # for ann in self.annotation:
        #     img_id = ann["image_id"]
        #     if img_id not in self.img_ids.keys():
        #         self.img_ids[img_id] = n
        #         n += 1

        self.prompts = [
            "A short image caption:",
            "A short image description:",
            "A photo of",
            "An image that shows",
            "Write a short description for the image.",
            "Write a description for the photo.",
            "Provide a description of what is presented in the photo.",
            "Briefly describe the content of the image.",
            "Can you briefly explain what you see in the image?",
            "Could you use a few words to describe what you perceive in the photo?",
            "Please provide a short depiction of the picture.",
            "Using language, provide a short account of the image.",
            "Use a few words to illustrate what is happening in the picture."
        ]

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        
        if 'caption' in ann.keys():
            text_output  = self.text_processor(ann["caption"])
        else:
            text_output = 'evaluation has no text output'

        choice = np.random.choice(len(self.prompts))

        text_input = self.prompts[choice]

        image_id = ann['image_id']
        
        return {
            "image": image,
            "text_input": text_input,
            #"image_id": self.img_ids[ann["image_id"]],
            'text_output': text_output,
            'image_id': image_id,
        }
    
    def collater(self, samples):
        image_list, question_list, answer_list, image_id_list = [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
           
            question_list.append(sample["text_input"])

            answers = sample["text_output"]

            answer_list.append(answers)
            
            image_id_list.append(sample['image_id'])
        

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
            'image_id': image_id_list,
        }


class CaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {
            "image": image,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }
    
class LLaVAPretrainDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        
        text_input  = self.text_processor(ann["text_input"]) 
        
        text_output = ann['text_output']
        
        return {
            "image": image,
            "text_input": text_input,
            'text_output': text_output,
        }
    
    def collater(self, samples):
        image_list, question_list, answer_list = [], [], []

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