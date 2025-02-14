from dataclasses import dataclass, field
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    PreTrainedTokenizer
)
from typing import Dict, List, Any

IGNORE_TOKEN_ID = -100  # 固定忽略token ID
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] }}{% elif message['role'] == 'user' %}{{ '\n<|user|>\n' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\n<|assistant|>\n' }}{% generation %}{{ message['content'] }}{{ eos_token }}{% endgeneration %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\n<|assistant|>\n' }}{% endif %}{% endfor %}"

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen-7B")

@dataclass
class DataArguments:
    train_data_path: str = field(metadata={"help": "训练数据路径"})
    eval_data_path: str = field(default=None, metadata={"help": "评估数据路径"})

@dataclass
class TrainingArguments(TrainingArguments):
    model_max_length: int = field(
        default=2048,
        metadata={"help": "最大序列长度（需适应显存）"}
    )

class ConversationDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.process_data(data)

    def process_data(self, raw_data: List[Dict]) -> List[Dict]:
        processed = []
        for item in tqdm(raw_data, desc="Processing data"):
            messages = item["messages"]
            # 使用模板生成输入序列和标签掩码
            tokenizer_output = self.tokenizer.apply_chat_template(
                messages,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
                return_assistant_tokens_mask=True,
                return_dict=True
            )
            input_ids = tokenizer_output["input_ids"][0]
            labels = input_ids.clone()
            assistant_masks = tokenizer_output["assistant_masks"]
            if isinstance(assistant_masks, list):
                assistant_masks = torch.tensor(assistant_masks)
            labels[assistant_masks == 0] = IGNORE_TOKEN_ID
            
            processed.append({
                "input_ids": input_ids,
                "attention_mask": tokenizer_output["attention_mask"][0],
                "labels": labels
            })
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_datasets(data_args: DataArguments, tokenizer: PreTrainedTokenizer, max_length: int):
    def load_jsonl(path: str):
        with open(path, 'r') as f:
            return [json.loads(line) for line in f]
    
    train_data = load_jsonl(data_args.train_data_path)
    eval_data = load_jsonl(data_args.eval_data_path) if data_args.eval_data_path else None
    
    return {
        "train": ConversationDataset(train_data, tokenizer, max_length),
        "eval": ConversationDataset(eval_data, tokenizer, max_length) if eval_data else None
    }

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 初始化模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        trust_remote_code=True
    )
    # if tokenizer.chat_template is None:
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 设置padding token

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        use_cache=False if training_args.gradient_checkpointing else True,  # model cache 与 gradient checkpointing 互斥
    )

    # 创建数据集
    data_module = create_datasets(data_args, tokenizer, training_args.model_max_length)

    # 配置Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_module["train"],
        eval_dataset=data_module["eval"],
        tokenizer=tokenizer
    )

    # 训练和保存
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    train()
