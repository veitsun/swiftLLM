import torch
import triton
import triton.language as tl

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.infer_state import LlamaInferState
from swiftllm.utils import cdiv

# 用于 prefill 解读那高效地将 KV 张量存储到分页 KV缓存中
# 这个内核的作用是，将那些刚刚计算出来的额 K 和 V 张量，高效，并行地复制并存储到正确的位置，非连续的物理内存块 （Paged KV Cache） 中
# 完成了从连续内存到非连续、分页式内存的关键数据拷贝任务，这是 vLLM 这类系统能够实现高吞吐量推理的重要基石之一
@triton.jit
def _fwd_kvcache_mgmt_prefill_kernel(
    k_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    v_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    k: torch.Tensor,	# [num_prefill_tokens, num_kv_heads, head_dim], contiguous 这是当前批次中所有序列的 Key 被拼接成了一个大的张量
    v: torch.Tensor,	# [num_prefill_tokens, num_kv_heads, head_dim], contiguous
    block_table: torch.Tensor,  # [*, max_blocks_per_seq], contiguous
    seq_ids: torch.Tensor,  # [num_prefill_seqs], contiguous
    prefill_seq_start_locs: torch.Tensor,  # [num_prefill_seqs], contiguous
    prefill_seq_lens: torch.Tensor,  # [num_prefill_seqs], contiguous
    cur_layer: int, # 表示当前正在处理的是模型的第几层，因为每一层都有自己独立的 KV 缓存，所以需要传入当前层的索引

    num_layers: tl.constexpr,
    num_kv_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
):
    # num_blocks 总的物理内存块数量
    # num_layers 模型层层数
    # num_kv_heads KV 注意力头数
    # block_size 每个块可以存储的 token 数量
    # head_dim 每个 KV 注意力头的维度

    # 并行化策略： 一个线程块处理一个序列的一个逻辑数据块

    # grid shape: [num_prefill_seqs, cdiv(max_prefill_len, block_size)]
    my_batch_id = tl.program_id(0)  # 序列索引
    my_block_id = tl.program_id(1)  # 块索引
    # 每个线程块处理一个序列的一个数据块

    # 边界检查 和 早期退出
    # 数据输入和偏移计算
    my_seq_len = tl.load(prefill_seq_lens + my_batch_id)
    my_seq_start_loc = tl.load(prefill_seq_start_locs + my_batch_id)  # 当前序列在全局 token 数组中的起始位置
    if my_block_id*block_size >= my_seq_len:
        return  # 当前块超出序列长度，直接返回
    
    # my_block_id*block_size 当前块在序列中的偏移
    # tl.arange(0, block_size).to(tl.int64) 块内的相对位置
    my_token_range = tl.arange(0, block_size).to(tl.int64) + my_block_id*block_size + my_seq_start_loc
    my_seq_id = tl.load(seq_ids + my_batch_id) # seq_ids + my_batch_id 是指针算术，指向数组中第 my_batch_id 个元素，就相当于取出那个prefill 序列的 id
    my_block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + my_block_id).to(tl.int64) # 取得块的索引

    # 多维偏移计算， 输入 KV 的偏移
    # 这里使用 triton 的广播机制，高效地计算出一个 [block_size, num_kv_heads, head_dim] 形状的偏移量张量，这正好对应要从 k 和 v 中加载的数据块的布局
    offs_kv = (my_token_range*num_kv_heads*head_dim).to(tl.int64)[:, None, None] + (tl.arange(0, num_kv_heads)*head_dim)[None, :, None] + tl.arange(0, head_dim)[None, None, :]
    # 同样使用广播机制计算目标地址。这个计算严格遵循了 k_cache 张量的 [num_blocks, num_layers, num_kv_heads, block_size, head_dim] 维度顺序，一步一步地定位到
    # 正确的物理块 my_block_index
    # 正确的层 cur_layer
    # 正确的头 tl.arange(0, num_kv_heads)
    # 正确块内 token 位置 (tl.arange(0, block_size))
    # 正确的头维度 tl.arange(0, head_dim)
    offs_kvcache = (my_block_index*num_layers+cur_layer)*num_kv_heads*block_size*head_dim + \
        (tl.arange(0, num_kv_heads)*block_size*head_dim)[None, :, None] + \
        (tl.arange(0, block_size)*head_dim)[:, None, None] + \
        tl.arange(0, head_dim)[None, None, :]
    
    # 这一步处理的是最后一个数据块的边界情况，如果序列长度不是 block_size 的整数倍
    # mask 会生成一个布尔张量，只有这些有效位置为 true，其余位置为 false
    mask = (my_token_range < my_seq_len + my_seq_start_loc)[:, None, None]
    tl.store(k_cache + offs_kvcache, tl.load(k + offs_kv, mask=mask), mask=mask)
    tl.store(v_cache + offs_kvcache, tl.load(v + offs_kv, mask=mask), mask=mask)

@triton.jit
def _fwd_kvcache_mgmt_decoding_kernel(
    k_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    v_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    k: torch.Tensor,	# [num_decoding_seqs, num_kv_heads, head_dim], contiguous
    v: torch.Tensor,	# [num_decoding_seqs, num_kv_heads, head_dim], contiguous
    block_table: torch.Tensor,  # [*, max_blocks_per_seq], contiguous
    decoding_seq_ids: torch.Tensor,  # [num_decoding_seqs], contiguous
    decoding_seq_lens: torch.Tensor,  # [num_decoding_seqs], contiguous
    cur_layer: int,

    num_layers: tl.constexpr,
    num_kv_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
):
    # grid shape: [num_decoding_seqs]
    my_batch_id = tl.program_id(0).to(tl.int64)
    my_seq_id = tl.load(decoding_seq_ids + my_batch_id)
    my_seq_len = tl.load(decoding_seq_lens + my_batch_id)
    my_block_id = (my_seq_len-1) // block_size
    my_block_offset = (my_seq_len-1) % block_size
    my_block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + my_block_id).to(tl.int64)

    offs_kv = my_batch_id*num_kv_heads*head_dim + (tl.arange(0, num_kv_heads)*head_dim)[:, None] + tl.arange(0, head_dim)[None, :]
    offs_kvcache = (my_block_index*num_layers+cur_layer)*num_kv_heads*block_size*head_dim + (tl.arange(0, num_kv_heads)*block_size*head_dim)[:, None] + my_block_offset*head_dim + tl.arange(0, head_dim)[None, :]

    tl.store(k_cache + offs_kvcache, tl.load(k + offs_kv))
    tl.store(v_cache + offs_kvcache, tl.load(v + offs_kv))

def store_kvcache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    model_config: LlamaModelConfig,
    engine_config: EngineConfig,
    infer_state: LlamaInferState,
    cur_layer: int
):
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert block_table.is_contiguous()
    assert infer_state.seq_ids.is_contiguous()
    assert infer_state.decoding_seq_lens.is_contiguous()

    if infer_state.num_prefill_seqs > 0:
        grid = (infer_state.num_prefill_seqs, cdiv(infer_state.max_prefill_len, engine_config.block_size))
        _fwd_kvcache_mgmt_prefill_kernel[grid](
            k_cache, v_cache,
            k, v,
            block_table,
            infer_state.seq_ids, infer_state.prefill_seq_start_locs, infer_state.prefill_seq_lens,
            cur_layer,
            model_config.num_layers, model_config.num_kv_heads, engine_config.block_size, model_config.head_dim, engine_config.max_blocks_per_seq
        )
        # max_blocks_per_seq = 2048, # 支持长达 32k tokens 的序列（每个 block 16 tokens）

    if infer_state.num_decoding_seqs > 0:
        grid = (infer_state.num_decoding_seqs,)
        _fwd_kvcache_mgmt_decoding_kernel[grid](
            k_cache, v_cache,
            k[infer_state.num_prefill_tokens:, :, :],
            v[infer_state.num_prefill_tokens:, :, :],
            block_table,
            infer_state.seq_ids[infer_state.num_prefill_seqs:],
            infer_state.decoding_seq_lens,
            cur_layer,
            model_config.num_layers, model_config.num_kv_heads, engine_config.block_size, model_config.head_dim, engine_config.max_blocks_per_seq
        )

        # for my_batch_id in range(infer_state.num_decoding_seqs):
        #     my_k = k[infer_state.num_prefill_tokens+my_batch_id]    # [num_kv_heads, head_dim]
        #     my_v = v[infer_state.num_prefill_tokens+my_batch_id]    # [num_kv_heads, head_dim]
        #     my_new_token_pos = infer_state.decoding_seq_lens[my_batch_id] - 1
        #     my_block_index = block_table[infer_state.seq_ids[infer_state.num_prefill_seqs+my_batch_id]][my_new_token_pos // engine_config.block_size]
        #     my_block_offset = my_new_token_pos % engine_config.block_size

        #     k_cache[my_block_index][cur_layer][:, my_block_offset, :] = my_k
        #     v_cache[my_block_index][cur_layer][:, my_block_offset, :] = my_v
