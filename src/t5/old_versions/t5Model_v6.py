import copy
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from transformers import T5Config, T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Block, T5LayerNorm

class T5Encoder(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.embed_tokens(input_ids)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        hidden_states = self.dropout(inputs_embeds)

        for i, layer_module in enumerate(self.block):

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                encoder_decoder_position_bias=None,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=None,
                use_cache=False,
                output_attentions=None,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class T5Decoder(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        if config.use_global:

            self.project1 = nn.Linear(config.d_model, config.d_model) # 768 * 768
            self.project2 = nn.Linear(config.d_model, config.d_model) # 768 * 768
            self.project3 = nn.Linear(config.d_model, config.d_model) # 768 * 768

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_attention_mask,
        attention_mask=None,
    ):

        use_global = self.config.use_global

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        attention_mask = torch.ones(batch_size, seq_length).to(inputs_embeds.device)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)

        hidden_states = self.dropout(inputs_embeds)

        for i, layer_module in enumerate(self.block):

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=None,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=None,
                use_cache=True,
                output_attentions=False,
            )

            hidden_states = layer_outputs[0]
            
            if use_global:

                if i == 8:
                        
                    logits = hidden_states.squeeze(1) # 10 * 768
                    # print(logits)

                    logits_attention = self.project1(logits) # 10 * 768
                    # print(logits_attention)

                    logits_attention = F.normalize(logits_attention)
                    # print(logits_attention)

                    logits_attention = torch.mm(logits_attention, logits.t()) # 10 * 10
                    # print(logits_attention)
                    
                    logits_attention = F.softmax(logits_attention, dim=1) # 10 * 10
                    # print(logits_attention)

                    logits_attention = torch.mm(logits_attention, logits) # 10 * 768
                    # exit()

                    hidden_states = logits_attention.unsqueeze(1)

                elif i == 9:

                    logits = hidden_states.squeeze(1)

                    logits_attention = self.project2(logits) # 10 * 768

                    logits_attention = F.normalize(logits_attention)

                    logits_attention = torch.mm(logits_attention, logits.t()) # 10 * 10
                    logits_attention = F.softmax(logits_attention, dim=1) # 10 * 10
                    logits_attention = torch.mm(logits_attention, logits) # 10 * 768

                    hidden_states = logits_attention.unsqueeze(1)

                if i == 10:
                    
                    logits = hidden_states.squeeze(1)

                    logits_attention = self.project3(logits) # 10 * 768

                    logits_attention = F.normalize(logits_attention)

                    logits_attention = torch.mm(logits_attention, logits.t()) # 10 * 10
                    logits_attention = F.softmax(logits_attention, dim=1) # 10 * 10
                    logits_attention = torch.mm(logits_attention, logits) # 10 * 768

                    hidden_states = logits_attention.unsqueeze(1)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class T5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"decoder\.project1\.weight",
        r"decoder\.project2\.weight",
        r"decoder\.project3\.weight",
        r"decoder\.project1\.bias",
        r"decoder\.project2\.bias",
        r"decoder\.project3\.bias",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        self.config = config

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Encoder(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Decoder(decoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Optional[torch.FloatTensor]:

        doc_size = self.config.doc_size

        encoder_outputs = self.encoder(
            input_ids=input_ids[:doc_size,:],
            attention_mask=attention_mask[:doc_size,:],
        )

        if self.config.grad_detach:

            with torch.no_grad():
                encoder_outputs_2 = self.encoder(
                    input_ids=input_ids[doc_size:,:],
                    attention_mask=attention_mask[doc_size:,:],
                )

            encoder_outputs = torch.cat([encoder_outputs, encoder_outputs_2], dim=0)

        decoder_input_ids=torch.zeros(encoder_outputs.size(0), 1, dtype=int).to(input_ids.device)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask[:encoder_outputs.size(0),:],
            attention_mask=None,
        )

        return decoder_outputs

class t5(nn.Module):
    def __init__(
        self,
        config: str,
        pretrained: str,
        num_global_layers:int,
        doc_size: int,
        use_global: bool,
        grad_detach: bool,
        encoder_only: bool=False,
        ) -> None:
        super(t5,self).__init__()
        self.config = T5Config.from_pretrained(config)

        self.config.num_global_layers = num_global_layers
        self.config.doc_size = doc_size
        self.config.use_global = use_global
        self.config.grad_detach = grad_detach

        self.t5 = T5Model.from_pretrained(pretrained, config=self.config)
        self.dense = nn.Linear(self.config.d_model, 2)


    def forward(self, input_ids, attention_mask):
        
        output= self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = output.squeeze()
        batch_score = self.dense(logits)

        return batch_score