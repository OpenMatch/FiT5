from transformers import T5Config, T5Model
from torch import nn
import torch

class t5(nn.Module):
    def __init__(
        self,
        pretrained: str
        ) -> None:
        super(t5,self).__init__()
        self.pretrained = pretrained
        self.config = T5Config.from_pretrained(self.pretrained)       
        self.t5 = T5Model.from_pretrained(self.pretrained, config=self.config)
        self.dense = nn.Linear(self.config.d_model, 1)

    def forward(self, input_ids, attention_mask):
        
        decoder_input_ids=torch.zeros(input_ids.size(0), 1, dtype=int).to(input_ids.device)

        output= self.t5(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
        )

        logits = output.last_hidden_state.squeeze()
        batch_score = self.dense(logits)

        return batch_score