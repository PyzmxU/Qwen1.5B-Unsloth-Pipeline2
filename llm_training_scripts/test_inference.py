import torch
import os

# 1. 终极补丁：解决 torchao 兼容性问题
for attr in ["int1", "int2", "int3","int4","int5","int6","int7", "uint1", "uint2", "uint3", "uint4", "uint5", "uint6", "uint7"]:
    if not hasattr(torch, attr):
        setattr(torch, attr, torch.int8)

from unsloth import FastLanguageModel

# 2. 加载你刚刚合并好的模型
# model_name 直接指向文件夹名字
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./qwen25_merged", 
    max_seq_length = 2048,
    load_in_4bit = False,  # 1.5B 模型 8G 显存直接全量加载即可，速度最快
    local_files_only = True,
)
FastLanguageModel.for_inference(model) # 开启推理加速

# 3. 定义你训练时使用的模板 (务必与训练保持一致)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# 4. 测试提问
instruction = "作为一名电子信息专业的学长，请给大一新生一条关于学习的建议。"
inputs = tokenizer(
[
    alpaca_prompt.format(instruction, "", "")
], return_tensors = "pt").to("cuda")

# 5. 执行生成
print("\n" + "="*30)
print("🤖 模型回答中...")
outputs = model.generate(**inputs, max_new_tokens = 512, temperature = 0.5)
result = tokenizer.batch_decode(outputs)

# 6. 提取 Response 之后的内容并打印
response_text = result[0].split("### Response:")[1].replace("<|endoftext|>", "").strip()
print(response_text)
print("="*30)