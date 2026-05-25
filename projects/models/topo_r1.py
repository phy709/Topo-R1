import torch
import torch.nn.functional as F
import contextlib
from .sa2va import Sa2VAModel


class VanillaDPOPolicy(Sa2VAModel):
    """
    普通的图像级直接偏好优化 (Vanilla Image-level DPO)
    作为消融实验的 Baseline，先对整图的 Log-Likelihood 求均值，再计算 Margin。
    """
    def __init__(self, dpo_beta=0.1, sft_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.dpo_beta = dpo_beta
        self.sft_weight = sft_weight
        
        # 挂载 LoRA
        if getattr(self.mllm, 'llm_lora_config', None) is not None:
            self.mllm.manual_prepare_llm_for_lora()
            print("[Aerial-Vanilla-DPO] LoRA Successfully Attached for Policy Model!")

    @property
    def _llm(self):
        if hasattr(self.mllm, 'llm'): return self.mllm.llm
        elif hasattr(self.mllm, 'model') and hasattr(self.mllm.model, 'language_model'): return self.mllm.model.language_model
        return self.mllm.model

    @contextlib.contextmanager
    def robust_disable_lora(self):
        """冻结 Adapter，瞬间将模型退化为 Reference Model"""
        if hasattr(self._llm, 'disable_adapter'):
            with self._llm.disable_adapter(): yield
        elif hasattr(self._llm, 'disable_adapters'):
            with self._llm.disable_adapters(): yield
        else: yield

    def _get_mask_logits(self, data, is_ref=False):
        """前向传播获取 SAM2 的预测掩码 Logits"""
        outputs = self.mllm(data.copy(), None, 'loss')
        hidden_states = outputs.hidden_states[-1]
        
        B = hidden_states.shape[0]
        seg_hiddens = torch.zeros((B, hidden_states.shape[-1]), device=hidden_states.device, dtype=hidden_states.dtype)
        for i in range(B):
            seg_idx = (data['input_ids'][i] == self.seg_token_idx).nonzero(as_tuple=True)[0]
            if len(seg_idx) > 0: seg_hiddens[i] = hidden_states[i, seg_idx[-1]]
                
        pred_embeddings = self.text_hidden_fcs(seg_hiddens)
        
        g_pixel_values = torch.stack([self.grounding_encoder.preprocess_image(p) for p in data['g_pixel_values']])
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=1)
        
        pred_masks = self.grounding_encoder.inject_language_embd(
            sam_states, pred_embeddings.unsqueeze(1), nf_nobj=(B, 1)
        ).flatten(0, 1)
        
        return pred_masks

    def forward(self, data, data_samples=None, mode='loss'):
        if mode != 'loss': return super().forward(data, data_samples, mode)

        # 数据拆包与缩放
        paired_masks = data.pop('masks')
        chosen_masks = [m[0:1] for m in paired_masks]    
        rejected_masks = [m[1:2] for m in paired_masks]  
        
        chosen_masks = torch.stack([F.interpolate(cm.unsqueeze(0).float(), size=(256, 256), mode='nearest').squeeze(0) for cm in chosen_masks]).squeeze(1)
        rejected_masks = torch.stack([F.interpolate(rm.unsqueeze(0).float(), size=(256, 256), mode='nearest').squeeze(0) for rm in rejected_masks]).squeeze(1)

        original_hs = getattr(self._llm.config, 'output_hidden_states', False)
        self._llm.config.output_hidden_states = True
        
        # ========================================================
        # 1. 计算 Reference Model 的 Log-Probs
        # ========================================================
        with torch.no_grad():
            with self.robust_disable_lora():
                ref_logits = self._get_mask_logits(data)
                
                # 形状变为 [B]，即每张图像只有一个总体的 Log-Prob
                ref_logprob_w = -F.binary_cross_entropy_with_logits(ref_logits, chosen_masks, reduction='none').mean(dim=(-1, -2))
                ref_logprob_l = -F.binary_cross_entropy_with_logits(ref_logits, rejected_masks, reduction='none').mean(dim=(-1, -2))
                
                del ref_logits
                torch.cuda.empty_cache()

        # ========================================================
        # 2. 计算 Policy Model 的 Log-Probs
        # ========================================================
        policy_logits = self._get_mask_logits(data)

        policy_logprob_w = -F.binary_cross_entropy_with_logits(policy_logits, chosen_masks, reduction='none').mean(dim=(-1, -2))
        policy_logprob_l = -F.binary_cross_entropy_with_logits(policy_logits, rejected_masks, reduction='none').mean(dim=(-1, -2))

        self._llm.config.output_hidden_states = original_hs

        # ========================================================
        # 3. 计算 Vanilla DPO 损失 (Image-level Margin)
        # ========================================================
        pi_ratio_w = policy_logprob_w - ref_logprob_w
        pi_ratio_l = policy_logprob_l - ref_logprob_l
        
        img_margin = pi_ratio_w - pi_ratio_l  # Shape: [B]

        loss_dpo = -F.logsigmoid(self.dpo_beta * img_margin).mean()

        loss_sft_bce = F.binary_cross_entropy_with_logits(policy_logits, chosen_masks)
        pred_sigmoid = torch.sigmoid(policy_logits)
        intersection = (pred_sigmoid * chosen_masks).sum(dim=(-1, -2))
        union = pred_sigmoid.sum(dim=(-1, -2)) + chosen_masks.sum(dim=(-1, -2))
        loss_sft_dice = (1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)).mean()
        
        loss_sft = self.sft_weight * (loss_sft_bce + 0.5 * loss_sft_dice)
        total_loss = loss_dpo + loss_sft

        return {
            'loss_dpo': loss_dpo,
            'loss_sft_chosen': loss_sft,
            'dpo_margin_mean': img_margin.mean().detach(), 
            'policy_chosen_logprob': policy_logprob_w.mean().detach(),
            'policy_rejected_logprob': policy_logprob_l.mean().detach(),
            'loss': total_loss
        }

class TopoDPOPolicy(Sa2VAModel):
    """
    基于底层视觉特征的直接偏好优化 (Mask-DPO)
    计算 Chosen 和 Rejected 掩码的 Log-Likelihood Margin。
    """
    def __init__(self, dpo_beta=0.1, sft_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.dpo_beta = dpo_beta
        self.sft_weight = sft_weight  # 在 DPO 中保留少量的 SFT Loss 防止灾难性遗忘
        
        # 挂载 LoRA
        if getattr(self.mllm, 'llm_lora_config', None) is not None:
            self.mllm.manual_prepare_llm_for_lora()
            print("[Aerial-DPO] LoRA Successfully Attached for Policy Model!")

    @property
    def _llm(self):
        if hasattr(self.mllm, 'llm'): return self.mllm.llm
        elif hasattr(self.mllm, 'model') and hasattr(self.mllm.model, 'language_model'): return self.mllm.model.language_model
        return self.mllm.model

    @contextlib.contextmanager
    def robust_disable_lora(self):
        """冻结 Adapter，瞬间将模型退化为 Reference Model"""
        if hasattr(self._llm, 'disable_adapter'):
            with self._llm.disable_adapter(): yield
        elif hasattr(self._llm, 'disable_adapters'):
            with self._llm.disable_adapters(): yield
        else: yield

    def _get_mask_logits(self, data, is_ref=False):
        """前向传播获取 SAM2 的预测掩码 Logits"""
        # 提取 Text Hidden States
        outputs = self.mllm(data.copy(), None, 'loss')
        hidden_states = outputs.hidden_states[-1]
        
        # 提取 [SEG] token 的 hidden state
        B = hidden_states.shape[0]
        seg_hiddens = torch.zeros((B, hidden_states.shape[-1]), device=hidden_states.device, dtype=hidden_states.dtype)
        for i in range(B):
            seg_idx = (data['input_ids'][i] == self.seg_token_idx).nonzero(as_tuple=True)[0]
            if len(seg_idx) > 0: seg_hiddens[i] = hidden_states[i, seg_idx[-1]]
                
        pred_embeddings = self.text_hidden_fcs(seg_hiddens)
        
        # 提取 SAM2 掩码
        g_pixel_values = torch.stack([self.grounding_encoder.preprocess_image(p) for p in data['g_pixel_values']])
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=1)
        
        pred_masks = self.grounding_encoder.inject_language_embd(
            sam_states, pred_embeddings.unsqueeze(1), nf_nobj=(B, 1)
        ).flatten(0, 1)
        
        return pred_masks

    def forward(self, data, data_samples=None, mode='loss'):
        if mode != 'loss': return super().forward(data, data_samples, mode)


        masks = data.pop('masks') 
        
        if isinstance(masks, list):
            # 若 collate 没自动堆叠 batch，这里是 list of [2, H, W]
            chosen_masks = [m[0:1] for m in masks]    # 切片获取第 0 层 -> [1, H, W]
            rejected_masks = [m[1:2] for m in masks]  # 切片获取第 1 层 -> [1, H, W]
        else:
            # 若 collate 自动堆叠成了 batch，这里是 [B, 2, H, W]
            chosen_masks = masks[:, 0:1]    # -> [B, 1, H, W]
            rejected_masks = masks[:, 1:2]  # -> [B, 1, H, W]

        chosen_masks = torch.stack([F.interpolate(cm.unsqueeze(0).float(), size=(256, 256), mode='nearest').squeeze(0) for cm in chosen_masks]).squeeze(1)
        rejected_masks = torch.stack([F.interpolate(rm.unsqueeze(0).float(), size=(256, 256), mode='nearest').squeeze(0) for rm in rejected_masks]).squeeze(1)

        # 强制开启 hidden_states 输出
        original_hs = getattr(self._llm.config, 'output_hidden_states', False)
        self._llm.config.output_hidden_states = True

        # ========================================================
        # 1. 错峰计算 Reference Model 的 Log-Probs (无梯度)
        # ========================================================
        with torch.no_grad():
            with self.robust_disable_lora():
                ref_logits = self._get_mask_logits(data)
                
                # 保留空间维度，得到形状为 [B, 256, 256] 的逐像素对数概率
                ref_logprob_w = -F.binary_cross_entropy_with_logits(ref_logits, chosen_masks, reduction='none')
                ref_logprob_l = -F.binary_cross_entropy_with_logits(ref_logits, rejected_masks, reduction='none')
                
                del ref_logits
                torch.cuda.empty_cache()

        # ========================================================
        # 2. 计算 Policy Model 的 Log-Probs (带梯度)
        # ========================================================
        policy_logits = self._get_mask_logits(data)
        
        # 同样保留空间维度 [B, 256, 256]
        policy_logprob_w = -F.binary_cross_entropy_with_logits(policy_logits, chosen_masks, reduction='none')
        policy_logprob_l = -F.binary_cross_entropy_with_logits(policy_logits, rejected_masks, reduction='none')

        self._llm.config.output_hidden_states = original_hs

        # ========================================================
        # 3. 计算 Dense DPO 损失 (Pixel-level Margin)
        # ========================================================
        # 因为前面没有求 mean，这里的运算全都是 [B, 256, 256] 形状的逐像素对抗！
        pi_ratio_w = policy_logprob_w - ref_logprob_w
        pi_ratio_l = policy_logprob_l - ref_logprob_l
        
        dense_margin = pi_ratio_w - pi_ratio_l  # Shape: [B, 256, 256]
        
        loss_dpo = -F.logsigmoid(self.dpo_beta * dense_margin).mean()

        # 附加项：微小的 SFT 约束，保证 Chosen 掩码本身画得准 (防止模型只顾拉开差距而画飞)
        loss_sft_bce = F.binary_cross_entropy_with_logits(policy_logits, chosen_masks)
        pred_sigmoid = torch.sigmoid(policy_logits)
        intersection = (pred_sigmoid * chosen_masks).sum(dim=(-1, -2))
        union = pred_sigmoid.sum(dim=(-1, -2)) + chosen_masks.sum(dim=(-1, -2))
        loss_sft_dice = (1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)).mean()
        
        loss_sft = self.sft_weight * (loss_sft_bce + 0.5 * loss_sft_dice)

        total_loss = loss_dpo + loss_sft

        return {
            'loss_dpo': loss_dpo,
            'loss_sft_chosen': loss_sft,
            'dpo_margin_mean': dense_margin.mean().detach(), # 这里取 mean() 回传给 Logger 记录，不影响梯度
            'policy_chosen_logprob': policy_logprob_w.mean().detach(),
            'policy_rejected_logprob': policy_logprob_l.mean().detach(),
            'loss': total_loss
        }