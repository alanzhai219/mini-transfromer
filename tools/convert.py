import torch
import safetensors
from safetensors.torch import save_file

import os
import json
import argparse

SUPPORTED_DTYPES = ["fp16"]

SUPPORTED_ARCHES = ["LlamaForCausalLM"]

string_dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "fp8": torch.float8_e5m2
}

class MetaData:
    def __init__(self, config, dtype):
        self.arch = config["architectures"][0]
        if self.arch not in SUPPORTED_ARCHES:
            raise Exception(f"Archiectre {self.arch} is not supported for now. arch must be in {SUPPORTED_ARCHES}")
        self.dtype = dtype
        if self.dtype not in SUPPORTED_DTYPES:
            raise Exception(f"Dtype {self.dtype} is not supported for now. arch must be in {SUPPORTED_DTYPES}")
        self.dim            = config["hidden_size"]
        self.hidden_dim     = config["intermediate_size"]
        self.head_dim       = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        self.n_layers       = config["num_hidden_layers"]
        self.n_heads        = config["num_attention_heads"]
        self.n_kv_heads     = config.get("num_key_value_heads", config["num_attention_heads"])
        self.vocab_size     = config["vocab_size"]
        self.max_seq_len    = config["max_position_embeddings"]
        self.bos_token_id   = config["bos_token_id"]
        self.eos_token_id   = config["eos_token_id"]
        self.rope_theta     = config.get("rope_theta", 100000)
        self.rotary_dim     = self.head_dim * config.get("partial_rotary_factor", 1)
        self.norm_eps       = config["rms_norm_eps"]
        self.norm_type      = "rmsnorm"
        self.use_cache      = config["use_cache"]

        assert config["hidden_act"] in ["gelu", "silu"]
        self.act_type       = config["hidden_act"]
        
        # TODO attention_bias, mlp_bias
        assert config.get("attention_bias", False) == False
        assert config.get("mlp_bias", False) == False
        # TODO moe
        if self.arch in ["MixtralForCausalLM"]:
            self.n_experts = config["num_local_experts"]
            self.n_experts_active = config["num_experts_per_tok"]

    def to_dict(self):
        res = {}
        res["arch"]         = self.arch
        res["dtype"]        = self.dtype
        res["dim"]          = str(self.dim)
        res["hidden_dim"]   = str(self.hidden_dim)
        res["head_dim"]     = str(self.head_dim)
        res["n_layers"]     = str(self.n_layers)
        res["n_heads"]      = str(self.n_heads)
        res["n_kv_heads"]   = str(self.n_kv_heads)
        res["vocab_size"]   = str(self.vocab_size)
        res["max_seq_len"]  = str(self.max_seq_len)
        res["bos_token_id"] = str(self.bos_token_id)
        res["eos_token_id"] = str(self.eos_token_id)
        res["rope_theta"]   = str(self.rope_theta)
        res["rotary_dim"]   = str(self.rotary_dim)
        res["norm_eps"]     = str(self.norm_eps)
        res["norm_type"]    = str(self.norm_type)
        res["act_type"]     = str(self.act_type)
        if self.arch in ["MixtralForCausalLM"]:
            res["n_experts"] = str(self.n_experts)
            res["n_experts_active"] = str(self.n_experts_active)
        return res

def load_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
        metadata = MetaData(config, args.dtype)
    return metadata

def load_tokenizer(tokenizer_file, vocab_size):
    with open(tokenizer_file, "r") as f:
        tokenizer = json.load(f)

    vocab = tokenizer["model"]["vocab"]
    assert len(vocab) <= vocab_size
    tokens = [""] * vocab_size
    for t, i in vocab.items():
        tokens[i] = t

    added_tokens = tokenizer["added_tokens"]
    for added in added_tokens:
        tokens[added["id"]]  = added["content"]

    for i, t in enumerate(tokens):
        t = t.replace('\u2581', ' ') # sentencepiece uses this character as whitespace
        b = t.encode('utf-8')
        b = b.replace(b"\0", b"\7") # replace null bytes with bell characters
        assert b.count(0) == 0
        tokens[i] = b
    
    return tokens

def permute_reverse(w, heads, rotary_dim):
    head_dim = w.shape[0] // heads
    assert rotary_dim <= head_dim
    w = torch.unflatten(w, 0, (-1, head_dim))

    # wr is rotray part, wk is kept unrotated
    wr = w[:, :rotray_dim]
    wk = w[:, rotary_dim:]
    
    # TODO

def load_weights(model_files, dtype_str, metadata, tie_word_embeddings=False):
    weights = {}
    for model_path in model_files:
        ext = os.path.splitext(model_path)[1]
        if ext == ".safetensors":
            with safetensors.safe_open(model_path, framework="pt") as f:
                for k in f.keys():
                    assert(k not in weights)
                    weights[k] = f.get_tensor(k)

    dtype = string_dtype_map[dtype_str]

    rotary_dim  = metadata.rotary_dim
    n_heads     = metadata.n_heads
    n_kv_heads  = metadata.n_kv_heads

    progress = 0;
    def cvt_type(t):
        nonlocal progress
        progress += 1
        print(f"\rConverting tensor {progress}: {t.shape}", end="", flush=True)
        return t.to(dtype)

    tensors = {}
    # convert embeding weight
    tensors["model.embed.weight"] = weights["model.embed_tokens.weight"].to(dtype)

    for l in range(metadata.n_layers):
        tensors[f"model.layers.{l}.attn.norm.weight"] = weights[f"model.layers.{l}.input_layernorm.weight"].float()

        # q
        tensors[f"model.layers.{l}.attn.wq.weight"] = weights[f"model.layers.{l}.self_attn.q_proj.weight"].to(dtype)
        # k
        tensors[f"model.layers.{l}.attn.wk.weight"] = weights[f"model.layers.{l}.self_attn.k_proj.weight"].to(dtype)
        # v
        tensors[f"model.layers.{l}.attn.wv.weight"] = weights[f"model.layers.{l}.self_attn.v_proj.weight"].to(dtype)
        # o
        tensors[f"model.layers.{l}.attn.wo.weight"] = weights[f"model.layers.{l}.self_attn.o_proj.weight"].to(dtype)

        # norm
        tensors[f"model.layers.{l}.mlp.norm.weight"] = weights[f"model.layers.{l}.post_attention_layernorm.weight"].float()

        tensors[f"model.layers.{l}.mlp.w1.weight"] = weights[f"model.layers.{l}.mlp.gate_proj.weight"].to(dtype)
        tensors[f"model.layers.{l}.mlp.w2.weight"] = weights[f"model.layers.{l}.mlp.down_proj.weight"].to(dtype)
        tensors[f"model.layers.{l}.mlp.w2.weight"] = weights[f"model.layers.{l}.mlp.up_proj.weight"].to(dtype)

    tensors[f"model.norm.weight"] = weights[f"model.norm.weight"].float()
    if tie_word_embeddings == False:
        tensors["model.output.weight"] = weights["lm_head.weight"].float()
    else:
        pass

    return tensors



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", type=str)
    parser.add_argument("-output", type=str)
    parser.add_argument("-dtype", type=str, default="fp16", choices=SUPPORTED_DTYPES)
    args = parser.parse_args()

    # define config.json, tokenizer.json, and *.safetensors 
    config_json_file = ""
    tokenizer_json_file = ""
    models_list = []
    if args.input is not None:
        # get config.json
        config_json_file = os.path.join(args.input, "config.json")
        if not os.path.exists(config_json_file):
            raise ValueError("Not config.json is found in {}".format(args.input))

        # get tokenizer.json
        tokenizer_json_file = os.path.join(args.input, "tokenizer.json")
        if not os.path.exists(tokenizer_json_file):
            raise ValueError("Not tokenizer.json is found in {}".format(args.input))

        # get models of .safetensors format
        files = os.listdir(args.input)
        for fname in files:
            if os.path.splitext(fname)[1] == ".safetensors":
                models_list.append(os.path.join(args.input, fname))
        if len(models_list) == 0:
            raise ValueError("No .safetensors are found in {}".format(args.input))
    else:
        raise ValueError("No input is specified!")

    # read config.json
    metadata = load_config(config_json_file)

    # read tokenizer.json
    tokens = load_tokenizer(tokenizer_json_file, metadata.vocab_size)

    # read *.safetensors
    tensors = load_weights(models_list, args.dtype, metadata)

    # addtional, add the token to the tensors in order to take good use of the whole file.
    concatenated_tensors = []
    for b in tokens:
        # Convert each token in the sequence to a list and append 0
        # token_list = [x for x in b] + [0]
        token_list = []
        for x in b:
            token_list.append(x)
        token_list.append(0)
        # Convert the list to a tensor with dtype uint8
        tensor = torch.tensor(token_list, dtype=torch.uint8)
        # Append the tensor to the list
        concatenated_tensors.append(tensor)
    
    # Concatenate all tensors in the list
    tensors["tokenizer.tokens"] = torch.cat(concatenated_tensors)

    # save model
    print(f"save model to {args.output}")
    save_file(tensors, args.output, metadata.to_dict())
