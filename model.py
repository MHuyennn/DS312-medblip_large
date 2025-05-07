import torch
import torch.nn as nn
import re

class FocalLoss(nn.Module):
    """Focal Loss cho bài toán multi-label classification."""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class ConceptHead(nn.Module):
    """Module tinh chỉnh đặc trưng hình ảnh cho concept detection."""
    def __init__(self, hidden_size, dropout_rate=0.3):
        super(ConceptHead, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln1 = nn.LayerNorm(hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln2(x)
        return x

class MedBLIPMultitask(nn.Module):
    def __init__(self, vision_encoder, text_decoder, processor):
        super(MedBLIPMultitask, self).__init__()
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.processor = processor
        self.concept_embeddings = None
        hidden_size = vision_encoder.config.hidden_size

        # Concept detection head
        self.concept_head = ConceptHead(hidden_size=hidden_size, dropout_rate=0.3)

        # Attention mechanism để cải thiện caption generation
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_size)

        # Dropout để tránh overfitting
        self.dropout = nn.Dropout(0.3)

        # Fine-tune một phần vision encoder (chỉ các tầng cuối)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.encoder.layers[-4:].parameters():  # Sửa từ layer thành layers
            param.requires_grad = True  # Fine-tune 4 tầng cuối

    def set_concept_embeddings(self, embeddings):
        """Set precomputed concept embeddings (Name embeddings)."""
        self.concept_embeddings = nn.Parameter(embeddings, requires_grad=False)

    def clean_caption(self, caption):
        """Xử lý hậu kỳ để làm sạch caption."""
        caption = caption.replace('clopsclops', '').replace('temperatures', '').replace('bravoliac', '')
        caption = re.sub(r'\b(\w+\s+\w+\s+)\1+', r'\1', caption)
        caption = re.sub(r'\s*-\s*', ' - ', caption)
        caption = re.sub(r'[^\w\s.,-]', '', caption)
        caption = ' '.join(caption.split())
        if caption and caption[-1] not in '.?!':
            caption += '.'
        return caption.strip()

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels_caption=None, mode="train"):
        # Encode image
        image_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_features = image_outputs.last_hidden_state[:, 0, :]  # CLS token cho concept
        vision_embeds = image_outputs.last_hidden_state  # Toàn bộ cho caption

        # Concept detection
        concept_features = self.concept_head(image_features)
        logits_concept = torch.matmul(concept_features, self.concept_embeddings.T)

        # Attention để tinh chỉnh vision embeds cho caption
        vision_embeds, _ = self.attention(
            vision_embeds, vision_embeds, vision_embeds,
            key_padding_mask=None
        )
        vision_embeds = self.attention_norm(vision_embeds + vision_embeds)  # Residual connection
        vision_embeds = self.dropout(vision_embeds)

        outputs = {}

        if mode == "train":
            # Caption Prediction
            caption_outputs = self.text_decoder(
                encoder_hidden_states=vision_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_caption
            )
            loss_caption = caption_outputs.loss

            outputs.update({
                "loss_caption": loss_caption,
                "logits_caption": caption_outputs.logits,
                "logits_concept": logits_concept
            })

        elif mode == "predict_caption":
            # Dự đoán caption
            generated_ids = self.text_decoder.generate(
                encoder_hidden_states=vision_embeds,
                max_length=200,
                num_beams=6,
                no_repeat_ngram_size=3,
                repetition_penalty=1.8,
                length_penalty=1.2
            )
            captions = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            captions = [self.clean_caption(caption) for caption in captions]
            outputs["generated_captions"] = captions

        elif mode == "predict_concept":
            outputs["logits_concept"] = logits_concept

        elif mode == "predict_both":
            # Dự đoán cả caption và concept
            generated_ids = self.text_decoder.generate(
                encoder_hidden_states=vision_embeds,
                max_length=200,
                num_beams=6,
                no_repeat_ngram_size=3,
                repetition_penalty=1.8,
                length_penalty=1.2
            )
            captions = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            captions = [self.clean_caption(caption) for caption in captions]

            outputs.update({
                "generated_captions": captions,
                "logits_concept": logits_concept
            })

        return outputs
