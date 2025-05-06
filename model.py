import torch
import torch.nn as nn
import re

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

    def clean_caption(self, caption):
        """Xử lý hậu kỳ để làm sạch caption."""
        # Loại bỏ cụm từ vô nghĩa
        caption = caption.replace('clopsclops', '').replace('temperatures', '').replace('bravoliac', '')
        # Loại bỏ lặp từ (ví dụ: "well - defined, well - defined" → "well - defined")
        caption = re.sub(r'\b(\w+\s+\w+\s+)\1+', r'\1', caption)
        # Chuẩn hóa dấu câu
        caption = re.sub(r'\s*-\s*', ' - ', caption)
        # Loại bỏ khoảng trắng thừa
        caption = ' '.join(caption.split())
        return caption.strip()

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels_caption=None, mode="train"):
        # Encode image (chỉ gọi vision encoder một lần)
        image_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_features = image_outputs.last_hidden_state[:, 0, :]  # CLS token features
        vision_embeds = image_outputs.last_hidden_state  # Dùng toàn bộ last_hidden_state cho caption

        outputs = {}

        if mode == "train":
            # Caption Prediction
            caption_outputs = self.text_decoder(
                encoder_hidden_states=vision_embeds,  # Dùng toàn bộ vision_embeds
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_caption
            )
            loss_caption = caption_outputs.loss

            # Concept Detection
            logits_concept = torch.matmul(image_features, self.concept_embeddings.T)  # Dot product dự đoán các concept

            outputs.update({
                "loss_caption": loss_caption,
                "logits_caption": caption_outputs.logits,
                "logits_concept": logits_concept
            })

        elif mode == "predict_caption":
            # Dự đoán caption
            generated_ids = self.text_decoder.generate(
                encoder_hidden_states=vision_embeds,
                max_length=100,
                num_beams=3,
                no_repeat_ngram_size=2,  # Ngăn lặp cụm từ
                repetition_penalty=1.2  # Phạt lặp từ
            )
            captions = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # Làm sạch captions
            captions = [self.clean_caption(caption) for caption in captions]
            outputs["generated_captions"] = captions

        elif mode == "predict_concept":
            # Dự đoán concept detection
            logits_concept = torch.matmul(image_features, self.concept_embeddings.T)
            outputs["logits_concept"] = logits_concept

        elif mode == "predict_both":
            # Dự đoán cả caption và concept
            generated_ids = self.text_decoder.generate(
                encoder_hidden_states=vision_embeds,
                max_length=100,
                num_beams=3,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2
            )
            captions = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            captions = [self.clean_caption(caption) for caption in captions]

            logits_concept = torch.matmul(image_features, self.concept_embeddings.T)

            outputs.update({
                "generated_captions": captions,
                "logits_concept": logits_concept
            })

        return outputs
