import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_rmsnorm(
	input_and_output: torch.Tensor,	# [num_tokens, hidden_size], contiguous
	weight: torch.Tensor,			# [hidden_size]
	eps: float,

	hidden_size: tl.constexpr
):
	# grid shape: [num_tokens]
	my_token_id = tl.program_id(0)
	input_and_output += my_token_id * hidden_size	# [hidden_size]

	offs = tl.arange(0, hidden_size)
	x = tl.load(input_and_output+offs).to(tl.float32)
	variance = tl.sum(x*x, axis=0) / hidden_size
	rstd = 1 / tl.sqrt(variance + eps)

	w = tl.load(weight+offs).to(tl.float32)
	x = x*rstd*w
	tl.store(input_and_output+offs, x.to(tl.float16))

def rmsnorm_inplace(
	input_and_output: torch.Tensor,	# [num_tokens, hidden_size]
	weight: torch.Tensor,
	eps: float
):
	grid = (input_and_output.shape[0], )
	_fwd_rmsnorm[grid](
		input_and_output,
		weight,
		eps,
		input_and_output.shape[1]
	)

# 这是一个用 triton 编写的 GPU Kernel，用于高效地执行融合地残差连接和 RMSNorm 操作。
@triton.jit
def _fwd_fused_add_rmsnorm(
	input_and_output: torch.Tensor,	# [num_tokens, hidden_size], contiguous
	residual_io: torch.Tensor,		# [num_tokens, hidden_size], contiguous
	weight: torch.Tensor,			# [hidden_size]
	eps: float,

	hidden_size: tl.constexpr
):
	# input_and_output: 输入输出张量，存储注意力层地输出
	# residual_io: 残差连接张量，用于残差连接
	# weight: RMSNorm 的缩放权重
	# eps: 数值稳定性参数

	# grid shape: [num_tokens]
	my_token_id = tl.program_id(0)  # 这里获取当前线程块的 id 作为 token索引，总共启动 num_tokens 个线程块
	input_and_output += my_token_id * hidden_size	# [hidden_size]
	residual_io += my_token_id * hidden_size

	# 加载数据
	offs = tl.arange(0, hidden_size)
	x = tl.load(input_and_output+offs) # 加载当前 token 特征
	r = tl.load(residual_io+offs)      # 加载残差特征
	# 残差连接
	x += r
	tl.store(residual_io+offs, x)      # 将结果存回 residual 缓冲区

	# RMS 归一化
	x = x.to(tl.float32)							# 转换为 float32 提高精度
	variance = tl.sum(x*x, axis=0) / hidden_size   # 计算方差
	rstd = 1 / tl.sqrt(variance + eps)				# 计算标准差的倒数，添加 eps 防止除零错误

	w = tl.load(weight+offs).to(tl.float32)	  # 加载归一化权重
	x = x*rstd*w															# 应用归一化
	tl.store(input_and_output+offs, x.to(tl.float16))

def fused_add_rmsnorm_inplace(
	input_and_output: torch.Tensor,	# [num_tokens, hidden_size]
	residual_io: torch.Tensor,
	weight: torch.Tensor,
	eps: float
):
	"""
	Perform fused add & rmsnorm

	This function accepts input_and_output (x), residual_io (r), and weight(w)
	as inputs, set r = x+r, and x = rms_norm(x+r, w)
	"""
	assert input_and_output.is_contiguous()
	assert residual_io.is_contiguous()
	assert weight.is_contiguous()
	grid = (input_and_output.shape[0], )
	_fwd_fused_add_rmsnorm[grid](
		input_and_output,
		residual_io,
		weight,
		eps,
		input_and_output.shape[1]
	)
