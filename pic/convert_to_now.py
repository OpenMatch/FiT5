import torch
import collections
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="5"
#old_pth = "/checkpoints_local/t5/v10_passage_global_cocondenser_retrain_0302/v10_passage_global_cocondenser_retrain_0302.bin_step-1500.bin"
old_pth = "/checkpoints_local/t5/v10_passage_global_cocondenser_100_no_add_token_3layer_0401/v10_passage_global_cocondenser_100_no_add_token_3layer_0401.bin_step-1500.bin"
state_dict = torch.load(old_pth, map_location='cuda:0')
#new_pth = "/checkpoints_local/t5/v10_passage_global_cocondenser_100_no_add_token_3layer_0401/v10_passage_global_cocondenser_100_no_add_token_3layer_0401.bin_step-1500.bin"
#new_state_dict = torch.load(new_pth)
new_state_dict = collections.OrderedDict()
for k, v in state_dict.items():
    if 'project' in k:
        print(k)
        # item = k.split('.')
        # id = item[3][len("project"):]
        # new_k_list = item[:3] + ["project_global", str(int(id)-1)] + item[4:]
        # new_k = ".".join(new_k_list)
        # new_state_dict[new_k] = v
    else:
        new_state_dict[k] = v

# for k, v in new_state_dict.items():
#     print(k)
# print(type(state_dict))
# print(type(new_state_dict))

# #new_old_pth = "/checkpoints_local/t5/v10_passage_global_cocondenser_retrain_0302/v10_passage_global_cocondenser_retrain_0302.bin_step-1500.bin.new"
# new_old_pth = "/checkpoints_local/t5/v10_passage_global_cocondenser_100_no_add_token_3layer_0401/v10_passage_global_cocondenser_100_no_add_token_3layer_0401.bin_step-1500.bin.new"
# torch.save(new_state_dict, new_old_pth)
# #new: module.t5.encoder.project_global.2.SelfAttention.v.weight
# #old: module.t5.encoder.project2.SelfAttention.v.weight