# 开源大语言模型监督式微调(SFT)实战指南

## 概述
本教程基于HuggingFace技术栈实现大语言模型的高效微调，支持以下工业级训练能力：
- 多GPU分布式训练（Accelerate）
- Zero3显存优化策略（DeepSpeed）
- 混合精度训练（FP16/BF16）
- 注意力计算加速（Flash Attention）


## 环境配置

> [!IMPORTANT]
> **CUDA版本要求**
> 本实验环境基于CUDA 12.4构建，验证命令：
> ```bash
> nvcc --version
> ```
> 若遇到段错误(segmentation fault)，请优先检查CUDA版本兼容性和各种CUDA相关环境变量配置

### 依赖管理方案
采用UV虚拟环境管理工具，比传统pip和conda快，[UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)：
```bash
# 安装UV工具链
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 核心依赖矩阵
| Library | Version | Functionality | Scope of Application |
|---------|---------|---------------|----------------------|
| `torch` | 2.5.1   | 张量计算与自动微分 | 所有模型操作的基础运行时 |
| `transformers` | 4.49.0.dev0 | 预训练模型架构 | 提供LLaMA-2、Mistral等SOTA模型支持 |
| `accelerate` | 1.3.0   | 分布式训练抽象层 | 统一单卡/多卡训练配置入口 |
| `deepspeed` | 0.15.4  | 显存优化策略 | 实现Zero3参数卸载与梯度累积 |
| `flash_attn` | 2.7.4   | 注意力机制优化 | 提升Transformer计算效率30%+ |

### 环境构建步骤
```bash
git clone git@github.com:ZZR0/llm-sft-demo.git

cd llm-sft-demo

# 创建隔离环境
uv venv llm-sft --python=3.11 && source llm-sft/bin/activate # 如果用fish，则使用llm-sft/bin/activate.fish

# 安装环境构建依赖
uv pip install setuptools wheel ninja

# 安装基础计算框架
uv pip install torch==2.5.1 transformers==4.48.3 --link-mode=copy

# 添加分布式支持
uv pip install accelerate==1.3.0 deepspeed==0.15.4 flash_attn==2.7.3 --link-mode=copy --no-build-isolation
```

### 安装验证步骤
```bash
# 检查核心库版本
python -c "import torch; print(torch.__version__)"  # 应输出2.5.1+cu124

# 测试CUDA可用性
python -c "import torch; print(torch.cuda.is_available())"  # 应输出True
```

## 数据集规范

### 数据格式标准
采用对话式JSONL格式，支持多轮对话场景：
```json
{
  "messages": [
    {"role": "system", "content": "你担任机器学习专家的角色"},
    {"role": "user", "content": "解释梯度消失问题"},
    {"role": "assistant", "content": "梯度消失常见于深层神经网络..."},
    {"role": "user", "content": "有哪些解决方案？"},
    {"role": "assistant", "content": "1. 使用ReLU激活函数\n2. 残差连接结构..."}
  ]
}
```

## 训练执行流程

### 1. 训练脚本创建
在项目根目录创建 `scripts/finetune.sh`：

```bash
#!/bin/bash
# 分布式训练启动脚本
# 使用说明：bash scripts/finetune.sh

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file scripts/zero3.yaml \
    scripts/finetune.py \
    --model_name_or_path /shd/zzr/models/qwen2.5-base-3b \
    --train_data_path data/train.jsonl \
    --eval_data_path data/eval.jsonl \
    --output_dir output/qwen2.5_base_3b_sft \
    --model_max_length 4096 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing true \
    --learning_rate 1e-5 \
    --bf16 true \
    --logging_dir output/qwen2.5_base_3b_sft/logs \
    --logging_steps 2 \
    --report_to none \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --seed 42
```

### 2. 配置文件说明
创建 `scripts/zero3.yaml` 配置DeepSpeed优化策略：
```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### 3. 训练执行命令
```bash
# 添加执行权限
chmod +x scripts/finetune.sh

# 启动分布式训练
scripts/finetune.sh

```

### 4. 核心参数分类说明

#### 模型配置组
| 参数 | 类型 | 默认值 | 示例值 | 作用说明 | 注意事项 |
|------|------|-------|--------|---------|----------|
| `--model_name_or_path` | string | 必填 | `/path/to/qwen2.5-base-3b` | 预训练模型路径 | 支持HuggingFace Hub ID或本地路径 |
| `--model_max_length` | int | 2048 | 4096 | 模型最大上下文长度 | 需与Flash Attention兼容 |
| `--gradient_checkpointing` | bool | false | true | 激活梯度检查点 | 节省30-50%显存，降低10%训练速度 |

#### 数据配置组
| 参数 | 类型 | 默认值 | 示例值 | 作用说明 | 注意事项 |
|------|------|-------|--------|---------|----------|
| `--train_data_path` | string | 必填 | `data/train.jsonl` | 训练集路径 | 支持单个文件 |
| `--eval_data_path` | string | 必填 | `data/eval.jsonl` | 验证集路径 | 建议保留5-10%数据 |

#### 训练循环控制
| 参数 | 类型 | 默认值 | 示例值 | 作用说明 | 计算公式 |
|------|------|-------|--------|---------|----------|
| `--num_train_epochs` | int | 3 | 5 | 训练轮次 | 总步数 = 样本数/(per_device_train_batch_size*gradient_accumulation_steps*GPU数)*epochs |
| `--per_device_train_batch_size` | int | 2 | 4 | 单GPU训练batch大小 | 实际batch = 该值 * gradient_accumulation_steps * GPU数 |
| `--per_device_eval_batch_size` | int | 2 | 8 | 单GPU验证batch大小 | 可设为训练batch的2-4倍 |
| `--gradient_accumulation_steps` | int | 16 | 32 | 梯度累积步数 | 等效增大batch_size倍数 |

#### 优化器配置
| 参数 | 类型 | 默认值 | 示例值 | 作用说明 | 调优建议 |
|------|------|-------|--------|---------|----------|
| `--learning_rate` | float | 1e-5 | 2e-5 | 初始学习率 | 7B模型建议1e-5 ~ 5e-5 |
| `--weight_decay` | float | 0.01 | 0.1 | L2正则化系数 | 防止过拟合重要参数 |
| `--max_grad_norm` | float | 1.0 | 0.5 | 梯度裁剪阈值 | 稳定训练关键参数 |

#### 日志与保存
| 参数 | 类型 | 默认值 | 示例值 | 作用说明 | 关联配置 |
|------|------|-------|--------|---------|----------|
| `--logging_steps` | int | 2 | 50 | 日志记录间隔 | 需配合--logging_dir使用 |
| `--save_strategy` | string | "steps" | "epoch" | 模型保存策略 | steps/epoch/no |
| `--save_steps` | int | 100 | 500 | 保存间隔步数 | 当策略为steps时生效 |
| `--save_total_limit` | int | 3 | 5 | 最大保存检查点数 | 自动删除旧检查点 |

#### 分布式训练
| 参数 | 类型 | 默认值 | 示例值 | 作用说明 | 底层实现 |
|------|------|-------|--------|---------|----------|
| `--bf16` | bool | true | false | Brain浮点加速 | 需要Ampere架构以上GPU |
| `--fp16` | bool | false | true | 传统混合精度训练 | 与bf16互斥 |

---

## 训练代码解析

以下是对`finetune.py`训练代码的详细解析，按照模块功能进行分章节说明：

### 整体结构说明
本代码基于Hugging Face Transformers库实现对话模型的微调，核心功能包括：
1. 参数配置管理（模型/数据/训练参数）
2. 对话数据处理与格式化
3. 自定义数据集构建
4. 模型加载与训练配置
5. 分布式训练执行

### 参数配置模块

1. 模型参数 `ModelArguments`
    - `model_name_or_path`：预训练模型名称或本地路径

2. 数据参数 `DataArguments`
    - 功能：定义数据路径
    - 参数说明：
    - `train_data_path`：必须参数，训练数据路径
    - `eval_data_path`：可选参数，验证数据路径

3. 训练参数 `TrainingArguments`
    继承自`transformers.TrainingArguments`，扩展模型长度参数：
    - 控制输入序列的最大长度
    - 需根据GPU显存调整

### 数据处理模块

#### 1. 对话模板 `DEFAULT_CHAT_TEMPLATE`
```jinja2
{% for message in messages %}
  {% if message.role == 'system' %}...{% endif %}
  {% if message.role == 'user' %}...{% endif %}
  {% if message.role == 'assistant' %}...{% endif %}
{% endfor %}
```
- 使用特殊token标记对话角色：
  - `<|system|>`：系统提示
  - `<|user|>`：用户输入
  - `<|assistant|>`：模型回复

#### 2. 对话模板应用示例

**原始对话数据**：
```python
messages = [
    {"role": "system", "content": "你是一个天气预报助手"},
    {"role": "user", "content": "今天北京天气如何？"},
    {"role": "assistant", "content": "北京今天晴，气温25-32℃，东南风2级"}
]
```

**模板处理后的文本**：
```
<|system|>
你是一个天气预报助手
<|user|>
今天北京天气如何？
<|assistant|>
{% generation %}
北京今天晴，气温25-32℃，东南风2级<|EOS|>
{% endgeneration %}
```

**实际分词结果**（假设）：
```
input_ids = [
    100,                    # <|system|>
    23456, 23457, 23458,    # "你是一个天气预报助手" 的分词
    101,                    # <|user|>
    34567, 34568, 34569,    # "今天北京天气如何？" 的分词
    102,                    # <|assistant|>
    45678, 45679, 45680,    # "北京今天晴..." 的分词
    103                     # <|EOS|>
]

assistant_mask = [
    0, 0, 0, 0, 0, 0, 0, 0  # system和user部分
    0, 1, 1, 1, 1           # assistant部分（不包括<|assistant|>标记）
]

labels = [
    -100, -100, -100, -100, -100, -100, -100, -100,  # 被忽略部分
    -100, 45678, 45679, 45680, 103  # 只保留助理回复的loss计算
]
```

该设计确保模型：
1. 仅在`<|assistant|>`后的生成位置进行预测
2. 自动处理多轮对话的上下文衔接
3. 正确处理生成结束标记`<|EOS|>`的位置

#### 2. 数据集类 `ConversationDataset`
核心处理逻辑在`process_data`方法：

```python
def process_data(self, raw_data):
    for item in tqdm(...):
        messages = item["messages"]
        tokenizer_output = self.tokenizer.apply_chat_template(
            messages,
            truncation=True,
            max_length=self.max_length,
            return_assistant_tokens_mask=True,
            ...
        )
        # 生成标签掩码
        labels = input_ids.clone()
        labels[assistant_masks == 0] = IGNORE_TOKEN_ID
```
- **关键步骤**：
  1. 使用`apply_chat_template`格式化对话
  2. 生成`assistant_masks`识别需要预测的部分
  3. 将非助理回复的标签设为`IGNORE_TOKEN_ID`（-100），在计算loss时忽略

### 模型初始化模块

#### 1. 分词器配置
```python
tokenizer = AutoTokenizer.from_pretrained(...)
tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```
- 关键配置：
  - 设置`padding_side="right"`保证生成方向正确
  - 当pad_token不存在时复用eos_token

#### 2. 模型加载
```python
model = AutoModelForCausalLM.from_pretrained(
    ...,
    torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    use_cache=not args.gradient_checkpointing
)
```
- 使用混合精度（bfloat16/float16）节省显存
- 梯度检查点与模型缓存互斥


### 训练执行流程

#### 1. 数据加载
```python
def load_jsonl(path):
    return [json.loads(line) for line in f]

train_data = load_jsonl(data_args.train_data_path)
```
- 输入数据格式要求：
  ```json
  {"messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]}
  ```

#### 2. Trainer配置
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=...,
    eval_dataset=...,
    tokenizer=tokenizer
)
```
- 自动支持：
  - 分布式训练
  - 混合精度训练
  - 模型保存与评估

#### 3. 训练执行
```python
trainer.train()  # 启动训练
trainer.save_model()  # 保存最终模型
```

### 关键实现细节

#### 1. 标签掩码机制
```python
labels[assistant_masks == 0] = IGNORE_TOKEN_ID
```
- 仅在助理回复位置计算loss
- 有效防止模型学习无关内容

#### 2. 内存优化策略
```python
use_cache=False if gradient_checkpointing else True
```
- 梯度检查点：用计算时间换显存空间
- 当启用时需关闭KV缓存

#### 3. 对话模板动态生成数据
```python
tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)
```
- 自动插入特殊token
- 生成助理token位置掩码

该实现完整展示了基于Hugging Face生态的对话模型微调流程，可根据具体需求调整模板和训练参数。

## 相关教程

本教材提供基础的对话模型微调实现，如需实现更复杂的训练功能或了解进阶技巧，推荐以下扩展资源：

### 1. Qwen2.5 微调指南
**链接**：https://github.com/QwenLM/Qwen2.5/blob/main/examples/llama-factory/finetune-zh.md

### 2. LLaMA-Factory 全功能训练框架
**链接**：https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md