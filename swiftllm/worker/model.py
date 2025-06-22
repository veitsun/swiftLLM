import itertools
import math

import torch

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import load_weights
from swiftllm.worker.block_manager import BlockManager
from swiftllm.utils import GB
import swiftllm_c

from .layers.pre_layer import LlamaPreLayer
from .layers.transformer_layer import LlamaTransformerLayer
from .layers.post_layer import LlamaPostLayer
from .infer_state import LlamaInferState

class LlamaModel:
    """
    LlamaModel - A Llama model that can be used for inference.

    This class also acts as a "worker" that resides on a particular GPU, waiting
    for the control plane (the scheduler) to send commands.

    To initialize, please:
    - call __init__()
    - call load_weights()
    - call profile_num_blocks() on one worker
    - call init_kvcache_and_swap()
    """

    @torch.inference_mode()
    def __init__(
        self,
        engine_config: EngineConfig
    ):
        """
        Initialize the LlamaModel.
        """
        self.engine_config = engine_config

        # Load model config
        self.model_config = LlamaModelConfig.load_from_model_path(engine_config.model_path)

        # Weight and RoPE cache
        self.weight = None
        self._cos_cached = self._sin_cached = None

        # Layers
        self.pre_layer = None
        self.transformer_layers = None
        self.post_layer = None

        # KV Cache
        self.num_blocks = None
        self.k_cache = self.v_cache = None
        self.k_swap = self.v_swap = None

        # Block manager
        self.cpu_block_manager = self.gpu_block_manager = None
        
    @torch.inference_mode()
    def load_weights(self):
        """
        Load weights and initialize layers
        """
        # Load weights
        self.weight = load_weights(
            self.model_config,
            torch.float16,
            self.engine_config.model_path,
            self.engine_config.use_dummy
        )

        # Initialize rotary embeddings
        self._init_to_get_rotary()

        # Initialize layers
        decoding_piggyback_stream = torch.cuda.Stream()
        self.pre_layer = LlamaPreLayer(self.model_config, self.weight)
        self.transformer_layers = [
            LlamaTransformerLayer(
                self.model_config,
                self.engine_config,
                self.weight.layers[layer_id],
                decoding_piggyback_stream,
                layer_id
            )
            for layer_id in range(self.model_config.num_layers)
        ]
        self.post_layer = LlamaPostLayer(self.model_config, self.weight)

    @torch.inference_mode()
    def profile_num_blocks(self) -> int:
        """
        Profiler the number of GPU blocks

        We run a forged prefill batch with the maximum number of tokens and
        sequences, record the peak memory usage, and infer the number of blocks
        that can be allocated.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Synthesis a prefill batch
        num_tokens = self.engine_config.max_tokens_in_batch
        batch_size = self.engine_config.max_batch_size
        input_lens = [num_tokens // batch_size] * batch_size
        input_lens[-1] += num_tokens % batch_size
        input_ids = [
            [0 for _ in range(input_len)]
            for input_len in input_lens
        ]
        seq_ids = list(range(batch_size))
        self.k_cache = self.v_cache = None # pylint: disable=attribute-defined-outside-init
        _ = self.forward(input_ids, seq_ids, [], ignore_kvcache=True)
        torch.cuda.synchronize()

        # peak_memory = torch.cuda.max_memory_allocated()
        # total_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory, total_memory = torch.cuda.mem_get_info()
        peak_memory = total_memory - free_memory
        useable_memory = total_memory*self.engine_config.gpu_mem_utilization
        print(f"[Model.profile] GPU total memory: {total_memory/GB:.2f} GB, runtime peak memory: {peak_memory/GB:.2f} GB")
        if useable_memory < peak_memory:
            raise RuntimeError(f"Peak memory {peak_memory/GB:.2f} GB exceeds usable memory {useable_memory/GB:.2f} GB ({total_memory/GB:.2f} GB * {self.engine_config.gpu_mem_utilization})")
        block_size_bytes = self.engine_config.block_size * self.model_config.get_kvslot_size()
        num_gpu_blocks = math.floor((useable_memory - peak_memory) / block_size_bytes)

        torch.cuda.empty_cache()
        return num_gpu_blocks
    
    @torch.inference_mode()
    def init_kvcache_and_swap(self, num_blocks: int):
        self.num_blocks = num_blocks

        # Initialize KV cache
        kvcache_shape = (
            self.num_blocks,
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.engine_config.block_size,
            self.model_config.head_dim
        )
        # Here we use torch.zeros instead of torch.empty, since that torch.empty
        # has the possibility to contain NaNs, which will cause the model to output NaNs.
        self.k_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")
        self.v_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")

        # Initialize KV swap space
        kvswap_shape = (
            self.engine_config.num_cpu_blocks,
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.engine_config.block_size,
            self.model_config.head_dim
        )
        self.k_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu")
        self.v_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu")

        # Initialize block manager
        self.gpu_block_manager = BlockManager(
            "GPU",
            self.num_blocks,
            self.engine_config.max_seqs_in_block_table,
            self.engine_config.max_blocks_per_seq,
            self.engine_config.block_size
        )
        self.cpu_block_manager = BlockManager(
            "CPU",
            self.engine_config.num_cpu_blocks,
            self.engine_config.max_seqs_in_block_table,
            self.engine_config.max_blocks_per_seq,
            self.engine_config.block_size
        )

    def _init_to_get_rotary(self):
        rope_scaling_factor = self.model_config.rope_scaling
        base = self.model_config.rope_theta
        max_position_embeddings = self.model_config.max_position_embeddings
        max_seq_len = max_position_embeddings * rope_scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, self.model_config.head_dim, 2, device="cuda", dtype=torch.float32) / self.model_config.head_dim))
        t = torch.arange(max_seq_len + 128, device="cuda", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float16)
        self._sin_cached = torch.sin(freqs).to(torch.float16)

    @torch.inference_mode()
    def _forward(
        self,
        input_ids: torch.Tensor,    # [total_token_num]
        infer_state: LlamaInferState,
    ) -> torch.Tensor:
        """
        Run a forward pass of the LlamaModel.
        """
        # 输入嵌入层处理，将 Token ids 转换为 embedding 向量
        # pre_layer 通常是词嵌入层，将离散的 token id 映射到连续的向量空间
        input_embds = self.pre_layer.forward(input_ids)
        
        # 创建一个与输入相同形状的零张量
        # 用于实现残差连接的优化版本，避免重复的内存分配
        residual_buf = torch.zeros_like(input_embds) # 这里只分配一次缓冲区，然后在遍历模型层的时候，重复使用这个缓冲区
        # transformer 层串联处理，这里逐层处理 transformer 层
        for layer in self.transformer_layers:
            input_embds = layer.forward(
                input_embds,
                residual_buf,
                self.k_cache,
                self.v_cache,
                self.gpu_block_manager.block_table if not infer_state.ignore_kvcache else None,  # 管理 GPU 内存中的 KV缓存块
                infer_state,
            )
        input_embds += residual_buf   # 进行残差连接，将累积的残差添加到最终的隐藏状态上， 这是一种优化的残差连接连接实现方式（之前是在 for 循环里面每次进行残差连接，现在是在for循环外面，进行累积的残差连接）
        output_tokens = self.post_layer.forward(input_embds, infer_state) # 将隐藏状态映射到词汇表大小的 logits， 返回每个位置上所有可能 token 的概率分布
        return output_tokens
    
    @torch.inference_mode()
    def forward(
        self,
        input_ids_list: list[list[int]], # [batch_size, *]，支持变长序列（每个序列有不同的长度）， 二维列表，外围列表长度是 batch_size，内部列表长度是每个序列的 token 数量，内层每个列表包含一个序列的 token ID
        seq_ids_list: list[int],     # [batch_size]，序列标识符
        decoding_seq_lens_list: list[int], # [num_decoding_seqs]， 解码序列长度（一个正在进行解码的序列长度），通过长度差计算出有多少个序列处于 prefill 阶段
        ignore_kvcache: bool = False,   # Skip actions related to kv cache, useful when profiling the number of kv blocks，是否忽略 KV 缓存（用于性能分析）
    ) -> list[int]:
        """
        Run a forward pass of the LlamaModel.

        This function is a wrapper of the `_forward` function. It prepares the infer_state
        and calls the `_forward` function.

        This function is intended to be called by the server.
        """
        # input_ids_list 中包含了所有序列的 Token ID 列表， 每个元素代表一个序列
        # Decoding_seq_lens_list 中只包含了解码阶段的长度序列
        # 这样确实可以计算出 prefill 阶段的序列数量
        num_prefill_seqs = len(input_ids_list) - len(decoding_seq_lens_list)
        # 数据扁平化处理
        # 将所有的输入序列连接成一个扁平数组，便于 GPU 批量处理
        flattened_input_ids = list(itertools.chain(*input_ids_list))
        # 序列长度管理，前半部分是 prefill 序列的实际长度，后半部分是 decoding 序列的长度
        # input_ids_list 中关于 解码的序列 可能只包含一个 token id

        """
        # 2 个 prefill 序列 + 1 个 decoding 序列
                input_ids_list = [
                    [1, 2, 3, 4],      # prefill 序列 1
                    [5, 6, 7],         # prefill 序列 2  
                    [8]                # decoding 序列(新生成的 token)
                ]
        """

        # input_ids_list[:num_prefill_seqs]
        # 获取前面 0 - num_prefill_seqs 个序列, 然后取出多个 prefill 序列 取其长度 作为新序列 然后再结合 decode 阶段的长度序列 
        # seq_lengths_list 在个代码中的作用可能是为了分配 KVCache 块
        seq_lengths_list = [len(seq) for seq in input_ids_list[:num_prefill_seqs]] + decoding_seq_lens_list
        """
            seq_lengths_list = [
                # Prefill 序列的长度（通过计算 input_ids_list 中序列长度得出）
                len(prefill_seq_1),
                len(prefill_seq_2),
                ...
                # Decoding 序列的长度(直接使用 decoding_seq_lens_list)
                decoding_seq_len_1,
                decoding_seq_len_2,
                ...
            ]
        """
        # seq_ids_list 是一个一维列表，包含了每个序列的 ID， 这里将其转变为张量
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        # 将 seq_ids_list 转换为 tensor，方便后续处理
        seq_lengths = torch.tensor(seq_lengths_list, dtype=torch.int32, device="cuda")

        batch_size = len(input_ids_list) # 一个批次处理多少个计算请求
        num_tokens = len(flattened_input_ids) # 一个批次的所有 token 数量

        # 提取所有 prefill 序列的长度信息，用于后续计算
        prefill_seq_lens_list = seq_lengths_list[:num_prefill_seqs]
        prefill_seq_lens = torch.tensor(prefill_seq_lens_list, dtype=torch.int32, device="cuda")
        # 计算每个 prefill 序列在扁平化数组中的起始位置
        # 计算 前缀和
        prefill_start_locs = torch.cumsum(prefill_seq_lens, dim=0, dtype=torch.int32) - prefill_seq_lens
        max_prefill_len = max(prefill_seq_lens_list) if prefill_seq_lens_list else 0
        """
            # 假设有 3 个 prefill 序列，长度分别为 [4, 3, 5]
            prefill_seq_lens = [4, 3, 5]

            # cumsum 计算累积和
            cumsum_result = [4, 7, 12]  # [4, 4+3, 4+3+5]

            # 减去原长度得到起始位置
            prefill_start_locs = [0, 4, 7]  # [4-4, 7-3, 12-5]
        """

        decoding_seq_lens = torch.tensor(decoding_seq_lens_list, dtype=torch.int32, device="cuda")
        max_decoding_len = max(decoding_seq_lens_list) if decoding_seq_lens_list else 0
        """
         max_prefill_len 和 max_decoding_len 分别是 prefill 阶段和 decoding 阶段的最大序列长度
         用于优化内存分配 和 并行处理
        """

        position_indices = torch.cat((
            torch.concat([
                torch.arange(
                    0,
                    prefill_seq_len,
                    device="cuda",
                    dtype=torch.int32
                )
                for prefill_seq_len in prefill_seq_lens_list
            ]) if prefill_seq_lens_list else torch.empty(0, device="cuda", dtype=torch.int32),
            decoding_seq_lens - 1
        ), dim=0)

        if not ignore_kvcache:
            # 告诉块管理器每个序列需要多少存储空间给 KVCache
            self.gpu_block_manager.allocate_blocks_for_seqs(
                seq_ids,
                seq_lengths
            )

        # Select the seq_block_size
        #
        # Here we use a simple heuristic:
        #
        # In paged attention phase 1, the grid shape is (num_decoding_seqs, num_kv_heads, cdiv(max_decoding_len, seq_block_size))
        # and among these blocks, num_kv_heads * sum(cdiv(decoding_seq_lens, seq_block_size)) blocks are useful.
        # Thus we set seq_block_size to be the largest integer that satisfies
        #      num_kv_heads * sum(cdiv(decoding_seq_lens, seq_block_size)) >= 1024
        # to fully utilize the GPU. Here 1024 is a magic number (since most high-end
        # GPUs have ~128 SMs, so ~512 SMSPs. Since the decoding-stage attention
        # is mostly a memory-bound operation, I think 1024 is a reasonable number.)
        #
        # In practice, we use `decoding_seq_lens_sum/seq_block_size` to approximate
        # sum(cdiv(decoding_seq_lens, seq_block_size))

        # 这段 while 循环是一个很聪明的启发式算法，它根据当前的工作负载动态调整 seq_block_size ，目标是让 GPU 上的活跃线程块数量接近 1024， 从而充分利用 GPU 的并行计算能力
        seq_block_size = 2048 # 这个块大小优化，是专门针对 decoding 阶段的
        decoding_seq_lens_sum = sum(decoding_seq_lens_list)
        # self.model_config.num_kv_heads * (decoding_seq_lens_sum / seq_block_size) 是计算活跃线程快递额数量
        while self.model_config.num_kv_heads*(decoding_seq_lens_sum/seq_block_size) < 1024 and seq_block_size//2 >= 64 and \
            max_decoding_len / (seq_block_size//2) <= 128:
            seq_block_size //= 2
        # seq_block_size // 2 >= 64 防止块太小导致计算效率低下
        # 用于传递推理状态
        infer_state = LlamaInferState(
            batch_size = batch_size,
            num_tokens = num_tokens,

            seq_ids = seq_ids,
            softmax_scale = self.model_config.head_dim ** -0.5,

            num_prefill_seqs = num_prefill_seqs,                # 所有 prefill 序列的数量
            num_prefill_tokens = num_tokens - (batch_size - num_prefill_seqs),
            prefill_seq_start_locs = prefill_start_locs,        # 每个序列的起始位置
            prefill_seq_start_locs_with_end = torch.cat([
                prefill_start_locs,
                torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
            ]),
            prefill_seq_lens = prefill_seq_lens,                # 每个 prefill 序列的长度
            max_prefill_len = max_prefill_len,                  # 所有 prefill 序列的最大长度

            num_decoding_seqs = batch_size - num_prefill_seqs,  # 所有 decoding 序列的数量
            decoding_seq_lens = decoding_seq_lens,              # 每个 decoding 序列的长度
            max_decoding_len = max_decoding_len,                # 所有 decoding 序列的最大长度

            seq_block_size = seq_block_size,
            num_seq_blocks = (max_decoding_len + seq_block_size-1) // seq_block_size,

            position_cos = self._cos_cached[position_indices],
            position_sin = self._sin_cached[position_indices],

            ignore_kvcache = ignore_kvcache
        )
        # 这些变量就像是 批处理的地图，告诉我们每个序列在哪里，有多长，以便正确高效地进行并行计算

        return self._forward(
            torch.tensor(flattened_input_ids, dtype=torch.int32, device="cuda"),
            infer_state
        ).tolist()

    def _swap(
        self,
        seq_ids_list: list[int],
        is_swap_in: bool
    ):
        src_block_manager = self.cpu_block_manager if is_swap_in else self.gpu_block_manager
        dst_block_manager = self.gpu_block_manager if is_swap_in else self.cpu_block_manager
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        seq_lengths = src_block_manager.get_num_allocated_blocks(seq_ids) * self.engine_config.block_size
        src_block_ids = src_block_manager.gather_allocated_blocks_and_free(seq_ids)
        dst_block_ids = dst_block_manager.allocate_blocks_for_seqs(seq_ids, seq_lengths)
        swiftllm_c.swap_blocks(
            src_block_ids.tolist(),
            dst_block_ids.tolist(),
            is_swap_in,

            self.k_cache, self.v_cache,
            self.k_swap, self.v_swap
        )
        
    @torch.inference_mode()
    def swap_in_seqs(
        self,
        seq_ids_list: list[int]
    ):
        """
        Swap in (move blocks from CPU to GPU) the specified sequences.
        """
        self._swap(seq_ids_list, True)
    
    @torch.inference_mode()
    def swap_out_seqs(
        self,
        seq_ids_list: list[int]
    ):
        """
        Swap out (move blocks from GPU to CPU) the specified sequences.
        """
        self._swap(seq_ids_list, False)

    @torch.inference_mode()
    def free_seqs_resources(self, seq_ids_list: list[int]):
        """
        Free the resources of the specified sequences.
        """
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        self.gpu_block_manager.free_blocks_for_seqs(seq_ids)
        self.cpu_block_manager.free_blocks_for_seqs(seq_ids)
