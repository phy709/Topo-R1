import os
import cv2
import copy
import numpy as np
import torch
import mmengine

from .base import Sa2VABaseDataset

class TopoDPODataset(Sa2VABaseDataset):
    """
    为 DPO 定制的 Dataset，同时加载 Chosen (GT) 和 Rejected (模型错题) 掩码
    """
    def __init__(self, data_root, ann_file, **kwargs):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.ann_file = ann_file
        self.single_image_mode = False
        self.data_list = self.load_data_list()

    def load_data_list(self):
        return mmengine.load(os.path.join(self.data_root, self.ann_file), file_format='json')

    def _parse_annotations(self, raw_ann_info):
        ann_info = copy.deepcopy(raw_ann_info)
        image_path = os.path.join(self.data_root, ann_info['image_path'])
        ann_info['image'] = image_path  

        # 1. 读取 Chosen Mask (Ground Truth)
        chosen_rel = ann_info.get('chosen_mask_path', None)
        if chosen_rel is not None:
            chosen_path = os.path.join(self.data_root, chosen_rel)
            chosen_img = cv2.imread(chosen_path, cv2.IMREAD_GRAYSCALE)
            if chosen_img is not None:
                chosen_bin = (chosen_img > 127).astype(np.uint8)
                ann_info['chosen_masks'] = torch.from_numpy(chosen_bin).unsqueeze(0)
            else:
                return None

        # 2. 读取 Rejected Mask (模型旧预测)
        rejected_rel = ann_info.get('rejected_mask_path', None)
        if rejected_rel is not None:
            rejected_path = os.path.join(self.data_root, rejected_rel)
            rejected_img = cv2.imread(rejected_path, cv2.IMREAD_GRAYSCALE)
            if rejected_img is not None:
                rejected_bin = (rejected_img > 127).astype(np.uint8)
                ann_info['rejected_masks'] = torch.from_numpy(rejected_bin).unsqueeze(0)
            else:
                return None

        return ann_info

    def prepare_data(self, index):
        raw_data_dict = self.data_list[index]
        data_dict = self._parse_annotations(raw_data_dict)
        if data_dict is None: return None

        out_data_dict = {}
        
        # 👇【核心修改】：打包偷渡！
        # data_dict['chosen_masks'] 是 [1, H, W]
        # data_dict['rejected_masks'] 是 [1, H, W]
        # 拼接后，out_data_dict['masks'] 变成 [2, H, W] 的张量
        out_data_dict['masks'] = torch.cat([data_dict['chosen_masks'], data_dict['rejected_masks']], dim=0)

        # 读取图片并处理 Token
        image = self._read_image(data_dict['image'])
        if image is None: return None
            
        image_data = self._process_single_image(image, self.single_image_mode)
        out_data_dict.update(image_data)
        
        prompt = data_dict.get('prompt', "<image>\nPlease segment it.")
        conversation = [{'from': 'human', 'value': prompt}, {'from': 'gpt', 'value': 'Sure. [SEG]'}]
        
        image_token_str = self._create_image_token_string(image_data['num_image_tokens'])
        conversation = self._process_conversations_for_encoding(conversation, image_token_str)
        token_dict = self.get_inputid_labels(conversation)
        out_data_dict.update(token_dict)
        
        return out_data_dict

    def real_len(self):
        return len(self.data_list)