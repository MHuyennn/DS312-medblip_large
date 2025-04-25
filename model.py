import torch
import torch.nn as nn
from transformers import BlipForConditionalGeneration

class MedBLIPMultiTaskModel(nn.Module):
    def __init__(self, num_concepts, pretrained_model="Salesforce/blip-image-captioning-large"):
        super(MedBLIPMultiTaskModel, self).__init__()
        # Caption backbone từ mô hình BLIP gốc
        self.blip = BlipForConditionalGeneration.from_pretrained(pretrained_model)

        # Freeze encoder nếu muốn (tuỳ chọn)
        # for param in self.blip.vision_model.parameters():
        #     param.requires_grad = False

        # Concept detection head: từ image features → multi-label vector
        self.classifier = nn.Sequential(
            nn.Linear(self.blip.vision_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_concepts)  # Output = số lượng CUI
        )

        self.loss_caption = nn.CrossEntropyLoss(ignore_index=self.blip.config.pad_token_id)
        self.loss_concept = nn.BCEWithLogitsLoss()  # Cho multi-label classification

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels_caption=None, labels_concept=None):
        # Encode ảnh
        encoder_outputs = self.blip.vision_model(pixel_values=pixel_values)

        # Dự đoán caption (decoder)
        outputs = self.blip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_caption,
            return_dict=True
        )

        # Lấy feature từ ảnh để dùng cho concept detection
        # Dùng feature trung bình toàn bộ patch (average pooling)
        last_hidden_state = encoder_outputs.last_hidden_state  # (B, N, H)
        pooled_image_feat = last_hidden_state.mean(dim=1)  # (B, H)

        logits_concept = self.classifier(pooled_image_feat)

        loss_caption = outputs.loss
        loss_concept = None
        total_loss = None

        if labels_concept is not None:
            loss_concept = self.loss_concept(logits_concept, labels_concept)
            total_loss = loss_caption + loss_concept
        else:
            total_loss = loss_caption

        return {
            "loss": total_loss,
            "loss_caption": loss_caption,
            "loss_concept": loss_concept,
            "logits_concept": logits_concept,
            "caption_output": outputs
        }
