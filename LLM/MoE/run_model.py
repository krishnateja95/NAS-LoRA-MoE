
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch

import os,sys

sys.path.append(".")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../../../..")

from utils.evaluate_ppl import eval_ppl

from models.modeling_qwen2_moe import Qwen2MoeForCausalLM

device_map = "auto"
# model_name = "mistralai/Mixtral-8x7B-v0.1"
model_name = "Qwen/Qwen1.5-MoE-A2.7B"

cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'
model_dir = "/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_dir,
                                            cache_dir   = cache_dir,
                                            torch_dtype = torch.float16,
                                            device_map  = device_map
                                            )
# del model

# model = AutoModelForCausalLM.from_config(config,
#                                          torch_dtype=torch.bfloat16,
#                                          trust_remote_code=True,
#                                         #  device_map="auto"
#                                          )

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side='left'

print(model)
model.seqlen = 2048
# eval_ppl_test(model=model, tokenizer=tokenizer)

print(eval_ppl(model, tokenizer, dataset = "wikitext2"))


