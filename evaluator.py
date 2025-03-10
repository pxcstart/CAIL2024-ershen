from transformers import AutoModelForCausalLM, AutoTokenizer
from metrics import BLEURT_score, Rouge_score
from peft import PeftModel
from tqdm import tqdm
import json
import torch


if __name__=="__main__":
    model = AutoModelForCausalLM.from_pretrained("../model/Qwen1.5-7B-Chat", torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, "./model/axolotl/checkpoint-488")
    tokenizer = AutoTokenizer.from_pretrained('../model/Qwen1.5-7B-Chat', padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    with open("./dataset/testset.json", 'r', encoding="utf-8") as file:
        testset = json.load(file)
    references = []
    for sample in testset:
        reference = sample[2]["content"]
        references.append(reference)
    with torch.no_grad():
        predictions = []
        for sample in tqdm(testset):
            q_input = tokenizer.apply_chat_template(sample[:2], tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([q_input], return_tensors="pt", truncation=True, max_length=2048, padding=False).to(model.device)
            outputs = model.generate(model_inputs.input_ids, max_new_tokens=512, temperature=0.7)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs)]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            predictions.append(response)
    result = []
    for id, (ref, prd) in enumerate(zip(references, predictions)):
        tmp = {}
        tmp["id"] = id
        tmp["reference"]=ref
        tmp["prediction"]=prd
        result.append(tmp)
        id += 1
    with open("./dataset/ft_result.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)
    BLEU = BLEURT_score(predictions, references)
    ROUGE_L = Rouge_score(predictions, references)
    print(f"BLEU Score:{BLEU};Rouge_L Score:{ROUGE_L}")