import copy
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from transformers import T5Config, T5PreTrainedModel
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5Block, T5LayerNorm

class T5Encoder(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens

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

        doc_size = self.config.doc_size

        # initialize past_key_values with `None` if past does not exist
        past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(None, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(None, self.config.num_layers)
        present_key_value_states = None
        all_hidden_states = None
        all_attentions = None
        all_cross_attentions = None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, layer_module in enumerate(self.block):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel

            layer_outputs = layer_module(
                hidden_states[:doc_size, :],
                attention_mask=extended_attention_mask[:doc_size, :],
                position_bias=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                encoder_decoder_position_bias=None,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=None,
                use_cache=False,
                output_attentions=False,
            )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            # append next layer key value states

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

class T5Decoder(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.use_global = self.config.use_global

        if self.use_global:
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
    ):

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.embed_tokens(input_ids)
        doc_size = self.config.doc_size

        batch_size, seq_length = input_shape

        attention_mask = torch.ones(batch_size, seq_length).to(inputs_embeds.device)

        # initialize past_key_values with `None` if past does not exist
        past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(None, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(None, self.config.num_layers)
        present_key_value_states = ()
        all_hidden_states = None
        all_attentions = None
        all_cross_attentions = None

        hidden_states = self.dropout(inputs_embeds)

        for i, layer_module in enumerate(self.block):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel

            layer_outputs = layer_module(
                hidden_states[:doc_size, :],
                attention_mask=extended_attention_mask[:doc_size, :],
                position_bias=None,
                encoder_hidden_states=encoder_hidden_states[:doc_size, :],
                encoder_attention_mask=encoder_extended_attention_mask[:doc_size, :],
                encoder_decoder_position_bias=None,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=None,
                use_cache=True,
                output_attentions=False,
            )

            hidden_states, present_key_value_state = layer_outputs[:2]

            if self.use_global:

                if i == 9:
                    
                    logits = hidden_states[:, 0, :]
                    logits_rest = hidden_states[:, 1:, :]

                    logits_attention = self.project1(logits).view(logits.size(0), -1) # 10 * 768
                    logits_attention = torch.mm(logits_attention, logits.t()) # 10 * 10
                    logits_attention = F.softmax(logits_attention, dim=1) # 10 * 10
                    logits_attention = torch.mm(logits_attention, logits) # 10 * 768

                    logits_attention = torch.cat([logits.unsqueeze(-1), logits_attention.unsqueeze(-1)], -1)
                    logits_attention = torch.matmul(logits_attention, torch.ones([2,1], device=logits.device))

                    hidden_states = torch.cat([logits_attention.view([logits_rest.size(0), 1, logits_rest.size(-1)]), logits_rest], 1)

                elif i == 10:

                    logits = hidden_states[:, 0, :]
                    logits_rest = hidden_states[:, 1:, :]

                    logits_attention = self.project2(logits).view(logits.size(0), -1) # 10 * 768
                    logits_attention = torch.mm(logits_attention, logits.t()) # 10 * 10
                    logits_attention = F.softmax(logits_attention, dim=1) # 10 * 10
                    logits_attention = torch.mm(logits_attention, logits) # 10 * 768

                    logits_attention = torch.cat([logits.unsqueeze(-1), logits_attention.unsqueeze(-1)], -1)
                    logits_attention = torch.matmul(logits_attention, torch.ones([2,1], device=logits.device))

                    hidden_states = torch.cat([logits_attention.view([logits_rest.size(0), 1, logits_rest.size(-1)]), logits_rest], 1)

                elif i == 11:
                    logits = hidden_states[:, 0, :]
                    logits_rest = hidden_states[:, 1:, :]

                    logits_attention = self.project3(logits).view(logits.size(0), -1) # 10 * 768
                    logits_attention = torch.mm(logits_attention, logits.t()) # 10 * 10
                    logits_attention = F.softmax(logits_attention, dim=1) # 10 * 10
                    logits_attention = torch.mm(logits_attention, logits) # 10 * 768

                    logits_attention = torch.cat([logits.unsqueeze(-1), logits_attention.unsqueeze(-1)], -1)
                    logits_attention = torch.matmul(logits_attention, torch.ones([2,1], device=logits.device))

                    hidden_states = torch.cat([logits_attention.view([logits_rest.size(0), 1, logits_rest.size(-1)]), logits_rest], 1)

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)

            # append next layer key value states
            present_key_value_states = present_key_value_states + (present_key_value_state,)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

class T5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        self.encoder = T5Encoder(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Decoder(decoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:

        batch_size = input_ids.size(0)

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = encoder_outputs[0]

        decoder_input_ids = torch.zeros(batch_size, 1, dtype=int).to(input_ids.device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class t5(nn.Module):
    def __init__(
        self,
        pretrained: str,
        doc_size: int,
        use_global: bool=False,
        grad_detach: bool=False,
        ) -> None:
        super(t5,self).__init__()
        self.pretrained = pretrained
        self.config = T5Config.from_pretrained(self.pretrained)
        self.config.doc_size = doc_size
        self.config.use_global = use_global
        self.config.grad_detach = grad_detach

        self.t5 = T5Model.from_pretrained(self.pretrained, config=self.config)
        self.dense = nn.Linear(self.config.d_model, 2)

    def forward(self, input_ids, attention_mask):
        
        assert input_ids.size(0) >= self.config.doc_size

        output= self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = output.last_hidden_state.squeeze()
        batch_score = self.dense(logits)

        return batch_score