from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, AdamW, BitsAndBytesConfig
from transformers.generation.utils import GenerationConfig
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import argparse
import torch
import json

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        prompt = f"<|im_start|>system\n{example[0]['content']}<|im_end|>\n<|im_start|>user\n{example[1]['content']}<|im_end|>\n<|im_start|>assistant\n"
        completion = f"{example[2]['content']}<|im_end|>"

        # Tokenize prompt and completion
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)

        # Combine prompt and completion
        input_ids = prompt_ids + completion_ids

        # Truncate or pad to max_length
        input_ids = input_ids[:self.max_length]
        labels = [-100] * len(prompt_ids) + completion_ids
        labels = labels[:self.max_length]

        # Ensure padding matches lengths
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        labels += [-100] * padding_length

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor([1] * (self.max_length - padding_length) + [0] * padding_length, dtype=torch.long)
        }

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="ChatLaw")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  
        llm_int8_enable_fp32_cpu_offload=True,
        bnb_4bit_compute_dtype=torch.bfloat16,#虽然我们以4位加载和存储模型，但我们在需要时会部分反量化他，并以16位精度进行计算
        bnb_4bit_quant_type="nf4",#nf量化类型
        bnb_4bit_use_double_quant=True,#双重量化
    )

    model = AutoModelForCausalLM.from_pretrained('../model/Qwen1.5-7B-Chat', torch_dtype=torch.float16, quantization_config=quantization_config, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained('../model/Qwen1.5-7B-Chat')
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=["gate_proj","up_proj","q_proj","down_proj","o_proj","k_proj","v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    with open("./dataset/trainset.json", "r", encoding="utf-8") as file:
        trainset = json.load(file)
    with open("./dataset/testset.json", "r", encoding="utf-8") as file:
        testset = json.load(file)
    references = []
    for sample in testset:
        reference = sample[2]["content"]
        references.append(reference)
    
    dataset = CustomDataset(trainset, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Set optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0 
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            loss = outputs.loss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss /= len(dataloader)/args.batch_size
        print(f"Epoch {epoch} Loss: {epoch_loss}")
    # Save model
    model.save_pretrained("./model/finetune")
    tokenizer.save_pretrained("./model/finetune")
