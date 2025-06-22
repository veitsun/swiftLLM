import torch

from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import LlamaWeight
from swiftllm.worker.kernels.rmsnorm import rmsnorm_inplace
from swiftllm.worker.infer_state import LlamaInferState
from swiftllm.worker.kernels.linear import linear

# 
class LlamaPostLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        weights: LlamaWeight,
    ):
        self.model_config = model_config
        self.weights = weights
    
    def forward(
        self,
        input_embds: torch.Tensor,	# [num_total_tokens, hidden_size]
        infer_state: LlamaInferState
    ) -> torch.Tensor:
        # Slice to get the last token embedding for each request
        # 输入 input_embeds 是一个扁平化的张量，包含了当前批次中所有序列，所有token的最终隐藏状态。需要从中准确地挑出每个序列的最后一个 token
        last_token_indices = torch.cat(
            (
                # Part A: 找出所有 prefill 序列的最后一个 token 的索引
                infer_state.prefill_seq_start_locs + infer_state.prefill_seq_lens - 1,
                # Part B: 找出所有 decode 序列的最后一个 token 的索引（在 input_embds 张量中，所有 decode 序列的 token（每个序列只有一个）被紧凑地排列在所有 prefill token 之后。）
                # 因此，它们的索引就是一个从 num_prefill_tokens 开始，到 num_tokens (总 token 数) 结束的连续整数序列。torch.arange 正是用于生成这样的序列。
                torch.arange(infer_state.num_prefill_tokens, infer_state.num_tokens, device=input_embds.device, dtype=torch.int32)
            ), dim=0
        ) # last_token_indices 包含了批次中每个序列最后一个 token 在 input_embds 中的索引位置
        # 通过 torch.cat 将这两部分索引拼接起来，last_token_indices 就成了一个完整的、按顺序排列的“最后 token 索引列表”。
        last_input = torch.empty((infer_state.batch_size, self.model_config.hidden_size), device=input_embds.device, dtype=input_embds.dtype)
        last_input[:, :] = input_embds[last_token_indices, :] # [batch_size, hidden_size]
        # Apply RMS-norm 最终层归一化
        rmsnorm_inplace(
            last_input,
            self.weights.final_norm,
            self.model_config.rms_norm_eps
        )
        # last_input 经过归一化的，代表每个序列下一个词元预测信息的张量
        # self.weights.lm_head 这是一个巨大的权重矩阵，维度通常是 [vocab_size, hidden_size]。在很多模型中，它与 token embedding 矩阵是共享权重的。
        # linear(...): 执行一次矩阵乘法。将 last_input 与 lm_head 权重矩阵相乘，得到 logits 张量。
        # logits 的维度是 [batch_size, vocab_size]。它的每一行代表一个序列，该行中的每个值对应词汇表中一个词的“得分”或“未归一化的对数概率”。
        logits = linear(last_input, self.weights.lm_head)    # [batch_size, vocab_size]
        # 预测输出 token
        output_tokens = torch.argmax(logits, dim=1) # 贪心解码，选择每个序列中 logits 最大的那个词元作为预测结果
        # dim=1: 表示沿着词汇表维度进行操作。
        # 对于 logits 中的每一行（每个序列），argmax 会找到得分最高的那个值的索引。
        # 这个索引，正好就是词汇表中对应 token 的 ID。
        # output_tokens: 最终返回的是一个一维张量，维度是 [batch_size]，其中每个元素就是对应序列预测出的下一个 token 的 ID。
        return output_tokens
    