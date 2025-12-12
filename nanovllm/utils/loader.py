"""
For some weights like qkv_proj in attention layer, we will pack multiple weights together for efficiency.
But in HF weights, they are separated tensors.
"""

import glob
import os
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.sampler import Sampler
from nanovllm.utils.context import set_context
import torch
from torch import nn
from safetensors import safe_open
import torch.distributed as dist
from transformers import AutoTokenizer


def default_weights_loader(params: nn.Parameter, weights: torch.Tensor) -> None:
    params.data.copy_(weights)


def load_model(model: nn.Module, path: str) -> None:
    """
    Arguments:
        model: the model to load weights into
        path: the path to the checkpoint
    Returns:
        None
    """
    packed_module_mapping = getattr(model, "packed_module_mapping", {})
    # use all tensor files in the checkpoint directory
    tensor_files = glob.glob(os.path.join(path, "*.safetensors"))
    for tensor_file in tensor_files:
        with safe_open(tensor_file, framework="pt", device="cpu") as f:
            # example weight_name: model.layers.0.self_attn.q_proj.weight
            for weight_name in f.keys():
                for short_weigth_name in packed_module_mapping:
                    if short_weigth_name in weight_name:
                        packed_weight_name, shard = packed_module_mapping[
                            short_weigth_name
                        ]
                        param_name = weight_name.replace(
                            short_weigth_name, packed_weight_name
                        )
                        param = model.get_parameter(param_name)
                        weights_loader = getattr(param, "weights_loader")
                        weights_loader(param, f.get_tensor(weight_name), shard)
                        break
                else:
                    # when not break above
                    param = model.get_parameter(weight_name)
                    weights_loader = getattr(
                        param, "weights_loader", default_weights_loader
                    )
                    weights_loader(param, f.get_tensor(weight_name))


if __name__ == "__main__":
    # Test parameters
    hidden_size = 1024
    intermediate_size = 3072
    total_num_heads = 16
    total_num_kv_heads = 8
    max_position_embeddings = 40960
    num_hidden_layers = 28
    vocab_size = 151936
    tie_word_embeddings = True
    head_dim = 128
    rope_theta = 1000000
    rms_norm_eps = 1e-6
    block_size = 256
    num_blocks = 10
    num_tokens = 4
    rank = 0
    world_size = 1

    # Initialize distributed process group
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://localhost:2333",
            rank=rank,
            world_size=world_size,
        )
    except RuntimeError:
        # Process group might already be initialized
        pass

    # Set default dtype and device
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")

    model = Qwen3ForCausalLM(
        num_hidden_layers,
        vocab_size,
        hidden_size,
        total_num_heads,
        total_num_kv_heads,
        max_position_embeddings,
        intermediate_size,
        head_dim=head_dim,
        tie_word_embeddings=tie_word_embeddings,
        rope_theta=rope_theta,
        rms_norm_eps=rms_norm_eps,
    )

    path = "/workspace/huggingface/Qwen3-0.6B/"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    prompt = "write me a haiku about the weather"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
    print(input_ids, input_ids.shape)
    input_ids = torch.tensor(input_ids).to(torch.int32)
    position_ids = torch.arange(len(input_ids)).to(torch.int32)
    # Setup context for attention
    kv_cache = torch.zeros(
        2,
        num_hidden_layers,
        num_blocks,
        block_size,
        total_num_kv_heads,
        hidden_size // total_num_kv_heads,
    )
    is_prefill = True
    cu_seqlens_q = torch.tensor([0, len(input_ids)], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, len(input_ids)], dtype=torch.int32)
    max_seqlen_q = len(input_ids)
    max_seqlen_k = len(input_ids)
    block_tables = None
    context_lens = torch.tensor([len(input_ids)], dtype=torch.int32)
    slot_mapping = torch.tensor(
        torch.arange(len(input_ids)),
        dtype=torch.int32,
    )
    set_context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        block_tables,
        context_lens,
        slot_mapping,
    )
    load_model(model, path)
    sampler = Sampler()
    with torch.no_grad():
        hidden_states = model(input_ids, position_ids)
        logits = model.compute_logits(hidden_states)
        sample_tokens = sampler(logits)
    print(sample_tokens, sample_tokens.shape)
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
