import torch

# --- 终极“欺骗”补丁：把所有可能的缺失属性全补上 ---
# 这样当 torchao 去找这些“未来”属性时，会拿到一个替代品，而不是直接崩溃
for attr in ["int1", "int2", "int3","int4","int5","int6","int7", "uint1", "uint2", "uint3", "uint4", "uint5", "uint6", "uint7"]:
    if not hasattr(torch, attr):
        # 统一指向 int8，这是目前最接近的合法类型
        setattr(torch, attr, torch.int8)
# ----------------------------------------------

# 补丁打完后再 import 其他库

import os

# --- 强制设置 HTTP/HTTPS 全局代理 ---
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"
# ----------------------------------
# import os

# # 1. 强行指定 NASM 路径 (假设你安装在默认位置)
# nasm_path = r"C:\Program Files\NASM"
# if nasm_path not in os.environ["PATH"]:
#     os.environ["PATH"] = nasm_path + os.pathsep + os.environ["PATH"]

# # 2. 告诉 CMake 汇编器就是 nasm
# os.environ["CMAKE_ASM_COMPILER"] = "nasm"

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. 核心显存保命参数配置
max_seq_length = 1024 # 限制序列长度，防止 KV Cache 撑爆 8G 显存
dtype = None
load_in_4bit = True   # 开启 4-bit 极致压缩

# 2. 加载基座模型 (选择 Qwen2.5-1.5B，轻量且强大，完美适配 8G)
model, tokenizer = FastLanguageModel.from_pretrained(
    #model_name = "unsloth/Qwen2.5-1.5B-Instruct",
    model_name = "./Qwen2.5-1.5B-Instruct",       # 改成这样！直接读取本地文件夹
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. 配置 LoRA 适配器 (指定要训练的网络层)
model = FastLanguageModel.get_peft_model(
    model,
    r = 8, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Unsloth 专属黑科技，省海量显存
    random_state = 3407,
)

# 4. 数据集格式化 (Alpaca 格式)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# 加载并映射我们刚刚建好的本地数据集
dataset = load_dataset("json", data_files="dataset.json", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 5. 配置训练器
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 1, # 显存底线
        gradient_accumulation_steps = 4, # 累计梯度模拟大 batch
        warmup_steps = 2,
        max_steps = 30,                  # Demo 测试，只跑 30 步看是否收敛
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",            # 8-bit 优化器，进一步省显存
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 6. 开始训练！
print("🚀 准备就绪，开始在 RTX 4060 上训练模型...")
trainer_stats = trainer.train()

# 7. 保存 LoRA 权重
model.save_pretrained("lora_model") 
tokenizer.save_pretrained("lora_model")
print("🎉 训练完成！你的专属 LoRA 权重已保存到 lora_model 文件夹！")
# 这一步不需要 C++ 编译器，也不需要 NASM
#model.save_pretrained_merged("qwen25_merged", tokenizer, save_method = "merged_16bit")
model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method = "q4_k_m")