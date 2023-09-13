"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from bliva.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from bliva.datasets.datasets.coco_caption_datasets import (
    COCOCapDataset,
    COCOCapEvalDataset,
    NoCapsEvalDataset,
)

from bliva.datasets.datasets.caption_datasets import TextCapsDataset, CaptionDataset, LLaVAPretrainDataset

from bliva.common.registry import registry
import os
import logging
import warnings

@registry.register_builder("llava_pretrain")
class LLaVAPretrainBuilder(BaseDatasetBuilder):
    train_dataset_cls = LLaVAPretrainDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/llava/pretrain_558.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
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
            ann_paths=[os.path.join(storage_path, 'blip_laion_cc_sbu_558k.json')], 
            vis_root=vis_root,
        )

        return datasets


@registry.register_builder("textcaps")
class TextCapsBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextCapsDataset
    #eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/textcaps/defaults.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
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
            ann_paths=[os.path.join(storage_path, 'TextCaps_0.1_train.json')], 
            vis_root=vis_root,
        )

        return datasets

@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    #eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
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
            ann_paths=[os.path.join(storage_path, 'coco_karpathy_train.json')], 
            vis_root=vis_root,
        )

        return datasets
    
@registry.register_builder("nocaps")
class NoCapBuilder(BaseDatasetBuilder):
    eval_dataset_cls =  CaptionDataset       #NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nocaps/defaults.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        vis_root = build_info.vis_root

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            ann_paths=[os.path.join(storage_path, 'nocaps_val_4500.json')], 
            vis_root=vis_root,
        )

        return datasets
    
@registry.register_builder("flickr30k")
class Flickr30kBuilder(BaseDatasetBuilder):
    eval_dataset_cls = CaptionDataset       #NoCapsEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/flickr30k/defaults.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage
        vis_root = build_info.vis_root

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.eval_dataset_cls
        datasets['eval'] = dataset_cls(
            vis_processor=self.vis_processors["eval"],
            text_processor=self.text_processors["eval"],
            ann_paths=[os.path.join(storage_path, 'flickr_val.json')], 
            vis_root=vis_root,
        )

        return datasets

