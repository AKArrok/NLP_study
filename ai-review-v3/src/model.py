import torch.nn as nn
from transformers import AutoModel

import config


def _resolve_pretrained_path():
    local_model_dir = config.PRE_TRAINED_DIR / 'bert-base-chinese'
    if local_model_dir.exists():
        return str(local_model_dir)
    return 'bert-base-chinese'


class ReviewAnalyzeModel(nn.Module):
    def __init__(self, freeze_bert=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(_resolve_pretrained_path())
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_output)
        return logits.squeeze(1)
