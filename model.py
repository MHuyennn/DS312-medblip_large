import torch
import torch.nn as nn
from transformers import BlipForConditionalGeneration

class MedBLIPMultitask(nn.Module):
    """Mô hình kết hợp BLIP cho Caption + Concept Detection song song."""

    def __init__(self, vision_encoder, text_decoder, processor, hidden_dim=768):
        super().__init__()
        self.processor = processor

        # Encoder hình ảnh BLIP
        self.vision_encoder = vision_encoder

        # Caption Decoder
        self.text_decoder = text_decoder

        # Concept Detection Head
        self.concept_projection = nn.Linear(hidden_dim, hidden_dim)
        self.concept_embeddings = None  # sẽ set sau bằng Name embeddings

    def set_concept_embeddings(self, embeddings):
        """Nhận embedding vector từ các concept Name."""
        self.concept_embeddings = embeddings  # (num_concept, hidden_dim)

    def forward(self, pixel_values, input_ids=None, attention_mask=None, mode="train"):
        # Trích xuất đặc trưng hình ảnh
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state[:, 0, :]  # lấy CLS token (batch_size, hidden_dim)

        outputs = {}

        # Caption prediction branch
        if mode in ["train", "caption"]:
            caption_outputs = self.text_decoder(
                encoder_hidden_states=image_embeds.unsqueeze(1),
                encoder_attention_mask=torch.ones(image_embeds.size(0), 1).to(image_embeds.device),
                labels=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            outputs["loss_caption"] = caption_outputs.loss
            outputs["logits_caption"] = caption_outputs.logits

        # Concept detection branch
        if mode in ["train", "concept"]:
            assert self.concept_embeddings is not None, "Concept embeddings chưa được set!"

            image_proj = self.concept_projection(image_embeds)  # (batch_size, hidden_dim)
            logits_concept = torch.matmul(image_proj, self.concept_embeddings.T)  # (batch_size, num_concepts)
            outputs["logits_concept"] = logits_concept

        return outputs
