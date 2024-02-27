import torch
from transformers import AutoModel
import copy

original_model = AutoModel.from_pretrained("t5-base")

print(original_model)

state_dict = original_model.state_dict()
keys = state_dict.keys()
print(keys)

new_state_dict = copy.deepcopy(state_dict)

for i in range(12):
    new_state_dict[f'encoder.block.{i}.layer.0.SelfAttention.o.weight'] /= 100
    new_state_dict[f'encoder.block.{i}.layer.1.DenseReluDense.wi.weight'] /= 10
    new_state_dict[f'encoder.block.{i}.layer.1.DenseReluDense.wo.weight'] /= 10

    new_state_dict[f'decoder.block.{i}.layer.1.EncDecAttention.o.weight'] /= 100
    new_state_dict[f'decoder.block.{i}.layer.0.SelfAttention.o.weight'] /= 100
    new_state_dict[f'decoder.block.{i}.layer.2.DenseReluDense.wi.weight'] /= 10
    new_state_dict[f'decoder.block.{i}.layer.2.DenseReluDense.wo.weight'] /= 10

new_state_dict['shared.weight'] /= 100


torch.save(new_state_dict, "/t5-base-fp16.bin")