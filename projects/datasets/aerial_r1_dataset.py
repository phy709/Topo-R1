import os
import cv2
import copy
import numpy as np
import torch
import mmengine

from .base import Sa2VABaseDataset

class AerialR1SFTDataset(Sa2VABaseDataset):
    def __init__(self, data_root, ann_file, **kwargs):
        # 初始化基类
        super().__init__(**kwargs)
        self.data_root = data_root
        self.ann_file = ann_file
        self.single_image_mode = False
        
        # ⚠️ 修复 1：手动加载数据并显式挂载到 self.data_list
        self.data_list = self.load_data_list()

    def load_data_list(self):
        annotations = mmengine.load(os.path.join(self.data_root, self.ann_file), file_format='json')
        return annotations

    def _parse_annotations(self, raw_ann_info):
        """
        解析逻辑：
        使用 deepcopy 防止在多 Epoch 循环时污染原始的 data_list 字典
        """
        ann_info = copy.deepcopy(raw_ann_info)
        
        # 补全图片的绝对路径
        image_path = os.path.join(self.data_root, ann_info['image'])

        mask_path_relative = ann_info.get('mask', None)
        if mask_path_relative is not None:
            # 分割任务：读取 PNG 掩码
            mask_path = os.path.join(self.data_root, mask_path_relative)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                print(f"[Warning] Failed to read mask, skipping: {mask_path}")
                return None
                
            # 严格的二值化防错，并转换格式
            binary_mask = (mask_img > 127).astype(np.uint8)
            masks = torch.from_numpy(binary_mask).unsqueeze(0)
            ann_info['masks'] = masks
        
        ann_info['image'] = image_path  
        return ann_info

    def prepare_data(self, index):
        # ⚠️ 修复 2：直接从 self.data_list 获取数据，彻底摆脱 super() 的束缚
        raw_data_dict = self.data_list[index]
        
        data_dict = self._parse_annotations(raw_data_dict)
        if data_dict is None:
            return None

        out_data_dict = {}
        if 'masks' in data_dict:
            out_data_dict['masks'] = data_dict['masks']

        # 读取图片
        image_file = data_dict['image']
        image = self._read_image(image_file)
        if image is None:
            return None
            
        # 喂给大模型做视觉编码
        image_data = self._process_single_image(image, self.single_image_mode)
        out_data_dict.update(image_data)
        
        # 处理多模态对话与 Token 化
        image_token_str = self._create_image_token_string(image_data['num_image_tokens'])
        conversation = self._process_conversations_for_encoding(data_dict['conversations'], image_token_str)
        token_dict = self.get_inputid_labels(conversation)
        out_data_dict.update(token_dict)
        
        return out_data_dict

    def real_len(self):
        return len(self.data_list)