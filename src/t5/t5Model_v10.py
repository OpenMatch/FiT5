import copy
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from transformers import T5Config, T5PreTrainedModel
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5Block, T5LayerNorm, T5LayerSelfAttention
from transformers.deepspeed import is_deepspeed_zero3_enabled
import math

class T5Encoder(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        if config.use_global:   # default num_global_layers=3
    
            #self.project1 = T5LayerSelfAttention(config) # 768 * 768
            #self.project2 = T5LayerSelfAttention(config) # 768 * 768
            #self.project3 = T5LayerSelfAttention(config) # 768 * 768
            self.project_global = nn.ModuleList([T5LayerSelfAttention(config) for i in range(config.num_global_layers)])
            self.global_layer_id = [config.num_layers - i for i in range(config.num_global_layers, 0, -1)]
            self.global_layer_id2projectid = {}
            for i in range(len(self.global_layer_id)):
                self.global_layer_id2projectid[self.global_layer_id[i]] = i

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
    
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, Attention, score_memory_func=None, score_memory=None, score_id=None, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = Attention.relative_attention_bias.weight.device
        if score_memory is not None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]#* score_memory
            memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]#* score_memory
            #context_position = torch.zeros(query_length, 1, dtype=torch.long, device=device)
            ##context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
            #memory_position = torch.zeros(1, key_length, dtype=torch.long, device=device)
            ##memory_position = torch.ones(1, key_length, dtype=torch.long, device=device) * score_memory
            #context_position[score_id][0] = score_memory
            #memory_position[0][score_id] = score_memory
        else:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
            memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]

        relative_position = memory_position - context_position #  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not Attention.is_decoder),
            num_buckets=Attention.relative_attention_num_buckets,
            max_distance=Attention.relative_attention_max_distance,
        )
        values = Attention.relative_attention_bias(relative_position_bucket) #score_memory_func(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        hidden_state_detach=None,
        score_token_ids=None,
        label_for_analyse=None,
        score_memory=None,
        score_memory_func=None,
    ):
        score_token_ids = None
        #print(score_memory)
        doc_size = self.config.doc_size
        use_global = self.config.use_global
        grad_detach = self.config.grad_detach
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)
        
        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
        
        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )
        if score_token_ids is not None:
            score_token_ids_mask = torch.ones([batch_size, self.config.num_layers, seq_length, seq_length], device=input_ids.device)
            #score_token_ids_mask[0, :, 0, :] = 0
            #score_token_ids_mask[0, :, :, 0] = 0
            
            for i, idx in enumerate(score_token_ids):
                score_token_ids_mask[i, :, idx, :] = 0
                score_token_ids_mask[i, :, :, idx] = 0
                

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        #if score_memory is not None and score_token_ids is not None:
        #    position_bias_list = []
        #    for i, o in enumerate(score_memory):
        #        position_bias_i = self.block[0].layer[0].SelfAttention.compute_bias(seq_length, seq_length) #self.compute_bias(seq_length, seq_length, self.block[0].layer[0].SelfAttention, score_memory_func, o, score_token_ids[i])
        #        position_bias_list.append(position_bias_i)
        #    position_bias = self.block[0].layer[0].SelfAttention.compute_bias(seq_length, seq_length) #torch.cat(position_bias_list, 0)
        #    #print(torch.sum(position_bias), seq_length, seq_length)
        #print(position_bias)
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        layer_score = None
        if label_for_analyse is not None:
            layer_score = {}
        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):

            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            else:
                layer_outputs = layer_module(
                    hidden_states[:doc_size,:,:],
                    attention_mask=extended_attention_mask[:doc_size,:,:],
                    position_bias=position_bias[:doc_size,:,:] if position_bias is not None else position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    #score_memory=score_memory, score_id=score_token_ids,
                )
                if grad_detach:
                    with torch.no_grad():

                        layer_outputs_detach = layer_module(
                            hidden_states[doc_size:,:,:],
                            attention_mask=extended_attention_mask[doc_size:,:,:],
                            position_bias=position_bias[doc_size:,:,:] if position_bias is not None else position_bias,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_extended_attention_mask,
                            encoder_decoder_position_bias=encoder_decoder_position_bias,
                            layer_head_mask=layer_head_mask,
                            cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                            past_key_value=past_key_value,
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                        )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]
            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            #if i == 0:
            #    if score_memory is not None and score_token_ids is not None:
            #        position_bias_list = []
            #        for i, o in enumerate(score_memory):
            #            position_bias_i = self.compute_bias(seq_length, seq_length, self.block[0].layer[0].SelfAttention, score_memory_func, o, score_token_ids[i])
            #            position_bias_list.append(position_bias_i)
            #        position_bias_score = torch.cat(position_bias_list, 0)
            #        position_bias += position_bias_score 
            #(batch_size, n_heads, seq_length, key_length) (100, 12, 200, 200)
            
            if score_token_ids is not None:
                position_bias = torch.mul(position_bias, score_token_ids_mask)
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            if grad_detach:

                if use_cache is False:
                    layer_outputs_detach = layer_outputs_detach[:1] + (None,) + layer_outputs_detach[1:]
                position_bias_detach = layer_outputs_detach[2]
                position_bias = torch.cat([position_bias, position_bias_detach], dim=0)
                hidden_states_detach, _ = layer_outputs_detach[:2]
                hidden_states = torch.cat([hidden_states, hidden_states_detach], dim=0)

            if use_global:
                n_doc = hidden_states.size(0)
                if i in self.global_layer_id:
                    #import pdb;pdb.set_trace()
                    project_global = self.project_global[self.global_layer_id2projectid[i]]
                    logits = hidden_states[:, 0, :].view(1, n_doc, -1) # 1 * 100 * 768/1024
                    logits_rest = hidden_states[:, 1:, :]

                    logits_attention = project_global(logits)[0] # 1 * 100 * 768/1024
                    logits_attention = logits + logits_attention

                    if label_for_analyse is not None:
                        score_norm = torch.nn.functional.normalize(logits_attention.view(100, -1), p=2, dim=1)
                        score_between_doc = torch.mm(score_norm, score_norm.transpose(0,1))
                        #import pdb;pdb.set_trace()
                        layer_score[str(i)] = {}
                        for rating_i in [0, 1, 2, 3]:
                            for rating_j in [0, 1, 2, 3]:
                                label_now_i = (label_for_analyse == rating_i).view(-1, 1) * 1.0
                                label_now_j = (label_for_analyse == rating_j).view(1, -1) * 1.0
                                between_doc = torch.mm(label_now_i, label_now_j)
                                between_doc_score = torch.sum(score_between_doc * between_doc) / torch.sum(between_doc)
                                layer_score[str(i)]["{}_{}_score".format(rating_i, rating_j)] = between_doc_score.item()
                    hidden_states = torch.cat([logits_attention.view(n_doc, 1, -1), logits_rest], 1)
                #torch.mm(logits_attention.view(100, -1),logits_attention.view(100, -1).transpose(0,1)).argmax(0)
                """if i == 9:

                    logits = hidden_states[:, 0, :].view(1, n_doc, -1) # 1 * 100 * 768
                    logits_rest = hidden_states[:, 1:, :]

                    logits_attention = self.project1(logits)[0] # 1 * 100 * 768
                    logits_attention = logits + logits_attention

                    hidden_states = torch.cat([logits_attention.view(n_doc, 1, -1), logits_rest], 1)

                elif i == 10:

                    logits = hidden_states[:, 0, :].view(1, n_doc, -1) # 1 * 100 * 768
                    logits_rest = hidden_states[:, 1:, :]

                    logits_attention = self.project2(logits)[0] # 10 * 768
                    logits_attention = logits + logits_attention

                    hidden_states = torch.cat([logits_attention.view(n_doc, 1, -1), logits_rest], 1)

                if i == 11:

                    logits = hidden_states[:, 0, :].view(1, n_doc, -1) # 1 * 100 * 768
                    logits_rest = hidden_states[:, 1:, :]

                    logits_attention = self.project3(logits)[0] # 10 * 768
                    logits_attention = logits + logits_attention

                    hidden_states = torch.cat([logits_attention.view(n_doc, 1, -1), logits_rest], 1)"""

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            ), layer_score
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        ), layer_score

class T5Decoder(T5PreTrainedModel):
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

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)
    

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
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
    
    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        model_embeds = self._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        self.vocab_size = new_num_tokens

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        # if word embeddings are not tied, make sure that lm head is resized as well
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            old_lm_head = self.get_output_embeddings()
            new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()
    
    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        if new_num_tokens is None:
            return old_embeddings

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
                old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that `old_embeddings` are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        else:
            new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        score_token_ids: Optional[torch.LongTensor] = None,
        label_for_analyse=None,
        score_memory=None,
        score_memory_func=None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        layer_score = None
        if encoder_outputs is None:
            encoder_outputs, layer_score = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                score_token_ids=score_token_ids,
                label_for_analyse=label_for_analyse,
                score_memory=score_memory,
                score_memory_func=score_memory_func,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        decoder_input_ids=torch.zeros(hidden_states.size(0), 1, dtype=int).to(input_ids.device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask[:hidden_states.size(0),:],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs, layer_score

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        ), layer_score

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
        new_tokenizer=None,
        score_memory=None,
        ) -> None:
        super(t5,self).__init__()
        self.config = T5Config.from_pretrained(config)

        self.config.num_global_layers = num_global_layers
        self.config.doc_size = doc_size
        self.config.use_global = use_global
        self.config.grad_detach = grad_detach

        self.t5 = T5Model.from_pretrained(pretrained, config=self.config)
        if new_tokenizer is not None:
            self.t5.resize_token_embeddings(len(new_tokenizer))
            self.config.vocab_size = len(new_tokenizer)
        
        self.dense = nn.Linear(self.config.d_model, self.config.vocab_size)

    def init_position(self, score_memory=None):
        self.position_memory = None
        if score_memory:
            self.position_memory = nn.Embedding(self.config.relative_attention_num_buckets, 12)
    def forward(self, input_ids, attention_mask, score_token_ids=None, label_for_analyse=None, score_memory=None):
        layer_score = None
        output, layer_score= self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            score_token_ids=score_token_ids,
            score_memory=score_memory,
            label_for_analyse=label_for_analyse,
            #score_memory_func=self.position_memory,
        )

        logits = output.last_hidden_state.squeeze()
        batch_score = self.dense(logits)
        if label_for_analyse is not None:
            return batch_score, layer_score
        else:
            return batch_score