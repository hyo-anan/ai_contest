import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig
from trl import SFTTrainer
from datasets import Dataset, DatasetDict
import json


# Model from Hugging Face hub
# yanolja/EEVE-Korean-10.8B-v1.0
# beomi/llama-2-koen-13b
# beomi/OPEN-SOLAR-KO-10.7B
base_model = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

# Quantization by 4bit
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Quantization the pre-trained model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Tokenizer load from model
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # 패딩토큰이 eos토큰인 부분(종료에 대한 학습인듯)
tokenizer.padding_side = "right"# 우측을 패딩(생성이 왼쪽에서부터 되니까)

print("모델 다운로드 완료")
# 학습 데이터 경로
train_data = "/workspace/jun4090/project/ai말평/train_data.json"
paths = [train_data]

inst_list = []
input_list = []
output_list = []

for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        # json 파일 읽기
        json_data = json.load(f)

    for i in json_data:
        inst_list.append(i["prompt"])
        input_list.append(i["input"])
        output_list.append(i["output"])

dict_data = {"instruction": inst_list, "input": input_list, "output": output_list}

# 데이터셋 생성
dataset = Dataset.from_dict(dict_data)

# datasetdict 생성
dataset_dict = DatasetDict({"train": dataset})
dataset_dict["train"] = dataset_dict["train"].shuffle(seed=42)

dataset = dataset_dict.map(
    lambda x: 
    {'text': f"###Instruction: 다음 대화를 요약해주세요.\n###Input: {x['input']}\n\n###Output: {x['output']}"}
)
dataset = dataset['train'].shuffle(seed=42)
# LoRA CONFIG
peft_params = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=16,
    target_modules=["q_proj","v_proj","k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# training args
training_params = TrainingArguments(
    output_dir=f"/workspace/jun4090/project/ai말평/result",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

# trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()
