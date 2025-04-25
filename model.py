import torch
import torch.nn as nn

class MedBLIPMultitask(nn.Module):
    def __init__(self, vision_encoder, text_decoder, processor):
        super(MedBLIPMultitask, self).__init__()
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.processor = processor
        self.concept_embeddings = None  # Sẽ set từ bên ngoài bằng set_concept_embeddings()

    def set_concept_embeddings(self, embedding_tensor):
        self.concept_embeddings = embedding_tensor  # (num_concepts, hidden_dim)

    def forward(self, pixel_values, input_ids=None, attention_mask=None, mode="caption"):
        image_embeds = self.vision_encoder(pixel_values=pixel_values).last_hidden_state[:, 0, :]  # CLS token

        if mode == "caption":
            decoder_output = self.text_decoder(input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               encoder_hidden_states=image_embeds.unsqueeze(1),
                                               encoder_attention_mask=torch.ones_like(pixel_values[:, 0, 0]))
            return decoder_output

        elif mode == "concept":
            if self.concept_embeddings is None:
                raise ValueError("Concept embeddings chưa được set!")
            logits = torch.matmul(image_embeds, self.concept_embeddings.T)  # (B, num_concepts)
            return logits
