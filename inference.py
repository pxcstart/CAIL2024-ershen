from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import json

Emb_model = SentenceTransformer("../model/m3e-base")
model = AutoModelForCausalLM.from_pretrained("../model/Qwen1.5-7B-Chat", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "./model/axolotl/checkpoint-488")
tokenizer = AutoTokenizer.from_pretrained('../model/Qwen1.5-7B-Chat', padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

system_prompt = """##指令：你是一个精通各种法律知识的法官，根据用户提供的案件基本信息和判决结果，以及相似案件的一审和二审判决结果，判断该案件的二审判决结果。"""

with open("./dataset/cleaned_queries.json", "r", encoding="utf-8") as file:
    queries = json.load(file)
with open("./dataset/cleaned_cases.json", "r", encoding="utf-8") as file:
    cases = json.load(file)
case_embs = torch.load("./dataset/case_embs.pt")

prediction = []
for sample in tqdm(queries): 
    background = [sample['yishen']['background']]
    emb = Emb_model.encode(background)
    emb = torch.tensor(emb).unsqueeze(0).to("cuda")
    distances = torch.cdist(emb, case_embs).squeeze(0)
    _, indices = torch.topk(-distances, k=3)
    indices = indices[0].tolist()

    query_input = f"""##输入：1.案件基本信息：{background}。\n 2.一审判决结果：{sample['yishen']['fact']}。\n \
    3.相似案件的判决结果：(1)<一审结果>：{cases[indices[0]]['yishen']['fact']}。<二审结果>：{cases[indices[0]]['ershen']['fact']}。\
    (2)<一审结果>：{cases[indices[1]]['yishen']['fact']}。<二审结果>：{cases[indices[1]]['ershen']['fact']}。\
    (3)<一审结果>：{cases[indices[2]]['yishen']['fact']}。<二审结果>：{cases[indices[2]]['ershen']['fact']}。\n ##输出："""
    messages = [{"role":"system", "content":system_prompt}, {"role":"user", "content":query_input}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        temperature=0.7
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    prediction.append({"background":sample["yishen"]['background'], 'yishen_fact':sample["yishen"]['fact'], "response":response})

with open("./dataset/inference.json", "w", encoding="utf-8") as file:
    json.dump(prediction, file, indent=4, ensure_ascii=False)

    

