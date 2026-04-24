import torch

# --- 补丁：防止 torchao 报错 ---
for attr in ["int1", "int2", "int3","int4","int5","int6","int7", "uint1", "uint2", "uint3", "uint4", "uint5", "uint6", "uint7"]:
    if not hasattr(torch, attr):
        setattr(torch, attr, torch.int8)
# ----------------------------

from unsloth import FastLanguageModel

# 1. 加载合并后的模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./qwen25_merged", # 指向你刚才保存的文件夹
    max_seq_length = 2048,
    load_in_4bit = False,         # 已经合并了，直接 16bit 加载即可
)
FastLanguageModel.for_inference(model) # 开启 2 倍速推理模式

# 2. 准备 Alpaca 格式的 Prompt
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# 3. 输入测试问题
instruction = "作为专业摄影师，请给我的松下 G9M2 推荐一个扫街的配置参数。"
inputs = tokenizer(
[
    alpaca_prompt.format(instruction, "", "")
], return_tensors = "pt").to("cuda")

# 4. 生成答案
print("🤖 模型正在思考...")
outputs = model.generate(**inputs, max_new_tokens = 512)
response = tokenizer.batch_decode(outputs)

# 5. 打印结果（截取 Response 之后的内容）
print(response[0].split("### Response:")[1].replace(" <|endoftext|>", ""))