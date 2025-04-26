import torch
import torch.nn as nn

class MedBLIPMultitask(nn.Module):
    def __init__(self, vision_encoder, text_decoder, processor):
        super(MedBLIPMultitask, self).__init__()
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.processor = processor
        self.concept_embeddings = None  # Khởi tạo chỗ để lưu embedding concept

    def set_concept_embeddings(self, embeddings):
        """Set precomputed concept embeddings (Name embeddings)."""
        self.concept_embeddings = nn.Parameter(embeddings, requires_grad=False)

    def forward(self, pixel_values, input_ids=None, attention_mask=None, mode="train"):
        # Encode image
        image_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_features = image_outputs.last_hidden_state[:, 0, :]  # CLS token features

        if mode == "train":
            # Caption Prediction
            caption_outputs = self.text_decoder(
                encoder_hidden_states=image_features.unsqueeze(1),  # Unsqueeze cho đúng shape
                input_ids=input_ids,  # Đây là chỗ quan trọng: truyền labels_caption làm input_ids
                attention_mask=attention_mask,
            )
            loss_caption = caption_outputs.loss

            # Concept Detection
            logits_concept = torch.matmul(image_features, self.concept_embeddings.T)  # Dot product dự đoán các concept

            return {
                "loss_caption": loss_caption,
                "logits_concept": logits_concept
            }

        elif mode == "predict_caption":
            # Dự đoán caption
            generated_ids = self.text_decoder.generate(
                encoder_hidden_states=image_features.unsqueeze(1),
                max_length=50,
                num_beams=3,
            )
            return generated_ids

        elif mode == "predict_concept":
            # Dự đoán concept detection
            logits_concept = torch.matmul(image_features, self.concept_embeddings.T)
            return logits_concept
