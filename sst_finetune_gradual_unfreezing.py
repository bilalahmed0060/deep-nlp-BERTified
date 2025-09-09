import torch
import torch.nn as nn
import torch.nn.functional as F
from bert import BertModel


class MultitaskBERTGradualFreeze(nn.Module):
    def __init__(self, config):
        super(MultitaskBERTGradualFreeze, self).__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=config.local_files_only
        )

        # Freeze all BERT params initially
        for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.sentiment_classifier = nn.Linear(config.hidden_size, config.N_SENTIMENT_CLASSES)

        # total layers from custom miniBERT
        self.total_layers = len(self.bert.bert_layers)
        self.unfrozen_layers = 0

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = outputs["pooler_output"]
        x = self.dropout(cls_embed)
        logits = self.sentiment_classifier(x)
        return logits

    def predict_sentiment(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask)

    def unfreeze_next_layer(self):
        if self.unfrozen_layers < self.total_layers:
            layer_idx = self.total_layers - self.unfrozen_layers - 1
            for param in self.bert.bert_layers[layer_idx].parameters():
                param.requires_grad = True
            self.unfrozen_layers += 1
            print(f"Unfroze BERT layer {layer_idx}")