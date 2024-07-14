import torch

d = torch.load("/data/users/pjlin/compacter/test_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-2_test/t5-base/mrpc/42/checkpoint-20/pytorch_model.bin")
print(d["prefix_shared"])


keys_to_delete = []
for key in d.keys():
    if key not in ["prefix_shared", "encoder.prefix_emb", "decoder.prefix_emb"]:
        keys_to_delete.append(key)
for key in keys_to_delete:
    del d[key]

print(d.keys())

torch.save(d, "/data/users/pjlin/compacter/test_outputs/prefix_shared.bin")
