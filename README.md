# Qwen1.5B-Unsloth-Pipeline2
一个实用的端到端 QLoRA 微调流水线，在 8GB RTX 4060 上运行 Qwen2.5-1.5B。包含处理 torchao bug 的运行时热补丁和直接 GGUF 导出。
# 🚀 Unsloth-8GB-VRAM-SFT: 极限显存下的微调与量化流水线

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Unsloth](https://img.shields.io/badge/Unsloth-QLoRA-orange.svg)](https://github.com/unslothai/unsloth)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--1.5B-green.svg)](https://huggingface.co/Qwen)

> **项目简介**：这是一个在消费级显卡（单卡 RTX 4060 8GB）上端到端跑通大语言模型（LLM）监督微调（SFT）的实战模板。项目重点不在于训练时长，而在于**验证微调流水线**，并提供了针对 Windows 本地环境常见底层依赖 Bug 的热修复（Hotfix）方案。

## ✨ 核心亮点 (Key Highlights)

本项目成功在 8GB 显存极限下完成了 Qwen2.5-1.5B 模型的微调，并展示了以下工程排障能力：

* **🐛 运行时底层热修复 (Runtime Hotfix)**：针对本地环境 `torchao` 库缺失 `int1~7` 等属性导致的系统级崩溃，利用 Python 反射机制 (`setattr`) 编写动态补丁，强制类型映射至 `int8`，成功绕过框架底层 Bug 阻塞。
* **⚡️ 极限显存压榨 (VRAM Optimization)**：综合运用 4-bit 量化加载 (NF4)、Unsloth 专属的 `gradient_checkpointing` 黑科技、以及 `adamw_8bit` 优化器，将训练显存死死压制在 8GB 物理红线内。
* **📦 无痛 GGUF 导出 (Bypassing Compilation)**：针对 Windows 环境下 16-bit 权重合并极易触发的 NASM / C++ 编译器路径缺失问题，果断切换技术路线，通过 Unsloth 原生支持直接将 LoRA 权重导出为 `q4_k_m` 格式的 GGUF 文件，实现极速本地部署。

---

## 🛠️ 快速开始 (Quick Start)

### 1. 环境依赖
推荐使用 Conda 管理环境。请确保已安装兼容的 CUDA 版本以及 Unsloth 框架：
```bash
pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
pip install --no-deps xformers trl peft accelerate bitsandbytes
