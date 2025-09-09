import torch
import torch.nn as nn
import torch.nn.functional as F
from bert import BertModel


class MultitaskBERTRDrop(nn.Module):

    def __init__(self, config):
        super(MultitaskBERTRDrop, self).__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=config.local_files_only
        )
        for param in self.bert.parameters():
            param.requires_grad = (config.option == "finetune")

        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.sentiment_classifier = nn.Linear(config.hidden_size, config.N_SENTIMENT_CLASSES)

    def forward_once(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = outputs["pooler_output"]  
        x = self.dropout(cls_embed)
        logits = self.sentiment_classifier(x)
        return logits

    def forward(self, input_ids, attention_mask):

        logits1 = self.forward_once(input_ids, attention_mask)
        logits2 = self.forward_once(input_ids, attention_mask)
        return logits1, logits2

    def predict_sentiment(self, input_ids, attention_mask):
        
        logits = self.forward_once(input_ids, attention_mask)
        return logits


def rdrop_loss(logits1, logits2, labels, alpha=1.0):
    ce_loss = 0.5 * (
        F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels)
    )

    log_p1 = F.log_softmax(logits1, dim=-1)
    log_p2 = F.log_softmax(logits2, dim=-1)
    p1 = F.softmax(logits1, dim=-1)
    p2 = F.softmax(logits2, dim=-1)

    kl_loss = 0.5 * (
        F.kl_div(log_p1, p2, reduction="batchmean") +
        F.kl_div(log_p2, p1, reduction="batchmean")
    )

    return ce_loss + alpha * kl_loss
