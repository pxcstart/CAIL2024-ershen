from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, AdamW, BitsAndBytesConfig
from transformers.generation.utils import GenerationConfig
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import argparse
import torch
import json


class FTDataset(Dataset):
    def __init__(self, tokenizer, dataset):
        super(FTDataset, self).__init__()
        self.tokenizer = tokenizer
        self.dataset = dataset

    def add_eos_token(self, result):
        result["input_ids"].append(self.tokenizer.eos_token_id)
        result["labels"].append(self.tokenizer.eos_token_id)
        result["attention_mask"].append(1)
        return result
    
    def list2tensor(self, result):
        for key, value in result.items():
            result[key] = torch.tensor(value)
        return result
    
    def tokenize(self, prompt, add_eos_token=False):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=2048,
            padding=False,
            return_tensors=None,
        )
        if (result["input_ids"][-1] != self.tokenizer.eos_token_id and len(result["input_ids"]) < 2048 and add_eos_token):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def qa_tokenize(self, message): # message格式：[{"role":user, "content":Q}, {"role":assistant, "content":A}]
        qa_input = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        qa_tokens = self.tokenize(qa_input, add_eos_token=True)
        q_input = self.tokenizer.apply_chat_template(message[:2], tokenize=False, add_generation_prompt=True)
        q_tokens = self.tokenize(q_input, add_eos_token=False)
        qa_tokens["labels"] = [-100] * len(q_tokens["input_ids"]) + qa_tokens["labels"][len(q_tokens["input_ids"]):]
        return self.list2tensor(qa_tokens)

    def __getitem__(self, index):
        sample = self.dataset[index]
        token_res = self.qa_tokenize(sample)
        return token_res
    
    def __len__(self):
        return len(self.dataset)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="ChatLaw")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
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
    
    train_dataset = FTDataset(tokenizer, trainset)
    # test_dataset = FTDataset(tokenizer, testset, "test")
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    # test_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collator)

    # 模型训练
    optimizer = AdamW(model.parameters(), lr=args.lr)
    result = []
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}/{args.epochs}", leave=False)
        epoch_loss = 0
        for step, inputs in enumerate(progress_bar):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            outputs = model(**inputs)
            loss = outputs.loss 
            epoch_loss += loss.item()
            if torch.isnan(loss):  # check
                breakpoint()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # 梯度裁剪
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(train_dataloader)}")
            
    model.save_pretrained("./model/finetune/")


    
