"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import logging
import warnings

from bliva.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from bliva.common.registry import registry
from bliva.datasets.datasets.aok_vqa_datasets import AOKVQADataset, AOKVQAEvalDataset, VQGAOKVQADataset
from bliva.datasets.datasets.coco_vqa_datasets import COCOVQADataset, COCOVQAEvalDataset, VQGCOCOVQADataset
from bliva.datasets.datasets.ocr_vqa_dataset import OCRVQADataset, STVQADataset, DocVQADataset
from bliva.datasets.datasets.llava_dataset import LLAVADataset


    
@registry.register_builder("ocrvqa") 
class OCRVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = OCRVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/ocrvqa/defaults.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        vis_root = build_info.vis_root

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'dataset.json')], 
            vis_root=vis_root,
        )

        return datasets
    


@registry.register_builder("textvqa") 
class TEXTVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = OCRVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/textvqa/defaults.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets... {}".format( self.__class__.__name__))
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        vis_root = build_info.vis_root

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'TextVQA_0.5.1_train.json')], 
            vis_root=vis_root,
        )

        return datasets


@registry.register_builder("coco_vqa")
class COCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_vqa.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets... {}".format( self.__class__.__name__))
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        vis_root = build_info.vis_root

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'vqa_train.json')], 
            vis_root=vis_root,
        )

        return datasets

@registry.register_builder("vqg_coco_vqa")
class VQGCOCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = VQGCOCOVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/vqg.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets... {}".format( self.__class__.__name__))
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        vis_root = build_info.vis_root

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'vqa_train.json')], 
            vis_root=vis_root,
        )

        return datasets


@registry.register_builder("ok_vqa")
class OKVQABuilder(COCOVQABuilder):
    train_dataset_cls = COCOVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/defaults.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets... {}".format( self.__class__.__name__))
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        vis_root = build_info.vis_root

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'okvqa_train.json')], 
            vis_root=vis_root,
        )

        return datasets
    
@registry.register_builder("vqg_ok_vqa")
class VQGOKVQABuilder(COCOVQABuilder):
    train_dataset_cls = VQGCOCOVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/vqg.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets... {}".format( self.__class__.__name__))
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        vis_root = build_info.vis_root

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'okvqa_train.json')], 
            vis_root=vis_root,
        )

        return datasets
    
    
@registry.register_builder("aok_vqa")
class AOKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset
    #eval_dataset_cls = AOKVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/aokvqa/defaults.yaml"}
    
    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets... {}".format( self.__class__.__name__))
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        vis_root = build_info.vis_root

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'aokvqa_v1p0_train.json')], 
            vis_root=vis_root,
        )

        return datasets

    
@registry.register_builder("vqg_aok_vqa")
class VQGAOKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = VQGAOKVQADataset
    #eval_dataset_cls = AOKVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/aokvqa/vqg.yaml"}
    
    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets... {}".format( self.__class__.__name__))
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        vis_root = build_info.vis_root

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'aokvqa_v1p0_train.json')], 
            vis_root=vis_root,
        )

        return datasets


@registry.register_builder("llavavqa") 
class LLAVABuilder(BaseDatasetBuilder):
    train_dataset_cls = LLAVADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/llava/defaults.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets... {}".format( self.__class__.__name__))
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        vis_root = build_info.vis_root

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'llava_instruct_extrapolated_2.json')], 
            vis_root=vis_root,
        )

        return datasets
    

@registry.register_builder("docvqa") 
class DocVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = DocVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/docvqa/defaults.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets... {}".format( self.__class__.__name__))
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        vis_root = build_info.vis_root

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train_v1.0.json')], 
            vis_root=vis_root,
        )

        return datasets
    
@registry.register_builder("stvqa") 
class STVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = STVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/stvqa/defaults.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets... {}".format( self.__class__.__name__))
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        vis_root = build_info.vis_root

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train_task_3.json')], 
            vis_root=vis_root,
        )

        return datasets