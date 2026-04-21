# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_internvl_chat import InternVLChatModel


class ModalityGate(nn.Module):
    """Lightweight per-image gating module."""

    def __init__(self, hidden_size: int, gate_hidden_ratio: float = 0.5):
        super().__init__()
        gate_hidden = max(1, int(hidden_size * gate_hidden_ratio))
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(pooled))


class InternVLChatModelGating(InternVLChatModel):
    """InternVLChatModel variant with per-image gating (no token count change)."""

    config_class = InternVLChatConfig

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config, vision_model=vision_model, language_model=language_model, use_flash_attn=use_flash_attn)

        hidden_size = config.llm_config.hidden_size
        self.enable_modality_gate = getattr(config, 'enable_modality_gate', True)
        self.gate_alpha = getattr(config, 'gate_alpha', 0.6)
        self.gate_center = getattr(config, 'gate_center', 0.8)
        gate_hidden_ratio = getattr(config, 'gate_hidden_ratio', 0.5)
        self.gate_module = ModalityGate(hidden_size, gate_hidden_ratio=gate_hidden_ratio)

    def load_gating_module(self, gate_module: nn.Module) -> None:
        """Load externally initialized gate module."""
        self.gate_module = gate_module

    def _apply_modality_gate(self, vit_embeds: torch.Tensor) -> torch.Tensor:
        if not self.enable_modality_gate or vit_embeds.numel() == 0:
            return vit_embeds
        pooled = vit_embeds.mean(dim=1)  # [num_images, hidden]
        gate = self.gate_module(pooled)  # [num_images, 1]
        # if dist.is_initialized() and dist.get_rank() == 0 and not hasattr(self, "_gate_logged"):
        #     with torch.no_grad():
        #         gate_values = gate.squeeze(-1).detach().cpu().tolist()
        #         print(f'gating values (per image): {gate_values}')
        #     self._gate_logged = True
        gate_values = gate.squeeze(-1).detach().cpu().tolist()
        print(f'gating values (per image): {gate_values}')
        scale = 1.0 + self.gate_alpha * (gate - self.gate_center)
        return vit_embeds * scale.unsqueeze(1)

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            statistics: Optional[torch.LongTensor] = None,
            loss_weight: Optional[List] = None,
            loss_reduction_all_gather: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_embeds = self._apply_modality_gate(vit_embeds)
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
            if statistics is not None:
                num_samples, num_padding_tokens, num_padding_images = statistics.tolist()
                self.num_samples += num_samples
                print(f'total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
            if ignore_flag:
                loss = loss * 0.0
        elif labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional["GenerationConfig"] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            vit_embeds = self._apply_modality_gate(vit_embeds)

            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
