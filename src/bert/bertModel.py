from re import S
from typing import Tuple

import torch
import torch.nn as nn

from transformers import BertConfig, BertModel

class bert(nn.Module):
    def __init__(
        self,
        pretrained: str,
        mode: str = 'cls',
        task: str = 'ranking'
    ) -> None:
        super(bert, self).__init__()
        self.pretrained = pretrained
        self.config = BertConfig.from_pretrained(self.pretrained)       
        self.bert = BertModel.from_pretrained(self.pretrained, config=self.config)
        self.dense = nn.Linear(self.config.hidden_size, 2)

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor = None, segment_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        output = self.bert(
            input_ids = input_ids.squeeze(), 
            attention_mask = input_mask.squeeze(),
            token_type_ids = segment_ids.squeeze(),
            # return_dict=False,
            )

        logits = output[0][:, 0, :] # 10 * 768
        batch_score = self.dense(logits).squeeze(-1) # 10 * 1

        return batch_score