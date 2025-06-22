import time
import argparse
from transformers import AutoTokenizer

import swiftllm
"""
这是一个使用 swiftllm 库进行大预言模型离线推理的示例脚本

"""

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.description = """
        An example script to demonstrate how to use the swiftllm model executor directly for inferencing without using the engine
    """
    parser.add_argument(
        "--model-path",
        help="Path to the model. Note: please download the model weights from HuggingFace in advance and specify the path here.",
        type=str,
        required=True
    )
    
    # 从命令行解析器中拿到 --model-path 参数
    model_path = parser.parse_args().model_path
    
    # 是整个 SwiftLLM 的核心配置环节
    engine_config = swiftllm.EngineConfig(
        model_path = model_path,
        use_dummy = False,
        
        block_size = 16, # 每个 block 的大小为 16 tokens
        gpu_mem_utilization = 0.99, # 使用 99% 的 GPU 内存
        num_cpu_blocks = 0, # 全部使用 GPU 进行计算，避免 CPU-GPU之间的传输延迟
        max_seqs_in_block_table = 128, # 最多同时处理 128 个序列
        max_blocks_per_seq = 2048, # 支持长达 32k tokens 的序列（每个 block 16 tokens）

        # The following are not used in the offline example
        max_batch_size = 16, # 最大批处理大小为 16
        max_tokens_in_batch = 2048*16 # 最大批处理中的 token 数量
    )

    start_time = time.perf_counter()

    # Initialize the model
    # For instructions on how to initialize the model, see comments in swiftllm/worker/model.py
    model = swiftllm.LlamaModel(engine_config) # 模型如何初始化，创建模型实例
    model.load_weights()                        # 加载模型权重
    num_blocks = model.profile_num_blocks() # 计算内存块数量
    print("Number of blocks:", num_blocks)
    model.init_kvcache_and_swap(num_blocks) # 内存如何分配，初始化 KV 缓存和交换

    model_creation_time = time.perf_counter() - start_time
    print(f"Model creation time: {model_creation_time:.2f} seconds") # 模型创建时间
    
    # 模型加载完成后，开始进行推理，首先准备测试数据
    prompts = [
        "Life blooms like a flower, far away",
        "one two three four five",
        "A B C D E F G H I J K L M N O P Q R S T U V",
        "To be or not to be,",
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    outputs = []

    # Prompt phase
    input_ids = tokenizer(prompts)['input_ids']
    # 推理时性能表现
    prompt_phase_outputs = model.forward(
        input_ids,
        list(range(0, len(prompts))),
        []
    )
    # print(tokenizer.batch_decode(prompt_phase_outputs, skip_special_tokens=True))
    outputs.append(prompt_phase_outputs)

    seq_lens = [len(x) for x in input_ids]
    last_round_outputs = prompt_phase_outputs

    # 逐步生成 token 阶段，固定生成 20 个 token ，在实际应用中通常会检查结束标记（EOS token） 或达到最大长度
    for _ in range(20):
        for i, _ in enumerate(prompts):
            seq_lens[i] += 1  # 更新每个序列的长度，方便 KVCache 管理

        # 模型前向推理
        last_round_outputs = model.forward(
            [[x] for x in last_round_outputs],  # 上一轮的输出作为这一轮的输入，每次只输入一个新的 token，而不是整个序列，保持与 prompt 阶段一致的输入格式
            list(range(0, len(prompts))),       # 序列索引
            seq_lens,                           # 每个序列的实时长度
        )
        # print(tokenizer.batch_decode(last_round_outputs, skip_special_tokens=True))
        outputs.append(last_round_outputs)
    
    # 输出处理
    for i, prompt in enumerate(prompts):
        output_tokens = [x[i] for x in outputs] # 获取每个 prompt 的输出 tokens
        output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
        print(f"{prompt}|{output_text}")
