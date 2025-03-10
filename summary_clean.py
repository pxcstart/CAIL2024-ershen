from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from tqdm import tqdm
import torch
import json

def SummaryChat(message, tokenizer, model):
    messages = [{"role": "system", "content":system_prompt},
                {"role": "user", "content":sample["yishen_content"]}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

if __name__=="__main__":
    model = AutoModelForCausalLM.from_pretrained('../model/Qwen1.5-7B-Chat', torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained('../model/Qwen1.5-7B-Chat', padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    system_prompt = "你是一个精通各种法律知识的法官，根据用户提供的案件陈述进行知识总结，同时满足以下要求：(1)总结要尽可能的精简准确,不要超过500字。(2)条理清晰,分点概括。(3)输出分为两部分，一部分为案件的基本信息，主要包括原告、被告、法院、上诉请求、事实和理由等信息；另一部分为法院判决结果。输出格式为`<案件基本信息>：`和`<法院判决结果>：`"
    
    with open("./origin_data/minicase.json", "r", encoding="utf-8") as file:
        cases = json.load(file)
    with open("./origin_data/queries.json", "r", encoding="utf-8") as file:
        queries = json.load(file)

    summary_cases = []
    for sample in tqdm(cases):
        one_messages = [{"role": "system", "content":system_prompt},
                    {"role": "user", "content":sample["yishen_content"]}]
        two_messages = [{"role": "system", "content":system_prompt},
                    {"role": "user", "content":sample["ershen_content"]}]
        response1 = SummaryChat(one_messages, tokenizer, model)
        response2 = SummaryChat(two_messages, tokenizer, model)
        summary_cases.append({"yishen":response1, "ershen":response2})

    summary_queries = []
    for sample in tqdm(queries):
        messages = [{"role": "system", "content":system_prompt},
                    {"role": "user", "content":sample["yishen_content"]}]
        response = SummaryChat(messages, tokenizer, model)
        summary_queries.append({"yishen":response})

    with open("./dataset/summary_cases.json", "w", encoding="utf-8") as file:
        json.dump(summary_cases, file, indent=4, ensure_ascii=False)
    with open("./dataset/summary_queries.json", "w", encoding="utf-8") as file:
        json.dump(summary_queries, file, indent=4, ensure_ascii=False)


# from transformers import AutoModelForCausalLM, AutoTokenizer
# from tqdm import tqdm
# import torch
# import json

# def batch_SummaryChat(messages_list, tokenizer, model):
#     """
#     处理一批数据，避免逐条调用 generate()
#     """
#     texts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in messages_list]
#     model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

#     with torch.inference_mode():  # 禁用梯度，提高推理速度
#         generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)

#     # 只取生成部分
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]
#     responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
#     return responses

# if __name__=="__main__":
#     model = AutoModelForCausalLM.from_pretrained('../model/Qwen1.5-7B-Chat', torch_dtype=torch.float16, device_map="auto")
#     tokenizer = AutoTokenizer.from_pretrained('../model/Qwen1.5-7B-Chat', padding_side="left")
#     tokenizer.pad_token = tokenizer.eos_token

#     system_prompt = "你是一个精通各种法律知识的法官，根据用户提供的案件陈述进行知识总结，同时满足以下要求：(1)总结要尽可能的精简准确,不要超过500字。(2)条理清晰,分点概括。(3)输出分为两部分，一部分为案件的基本信息，主要包括原告、被告、法院、上诉请求、事实和理由等信息；另一部分为法院判决结果。输出格式为`<案件基本信息>：`和`<法院判决结果>：`"

#     with open("./origin_data/minicase.json", "r", encoding="utf-8") as file:
#         cases = json.load(file)
#     with open("./origin_data/queries.json", "r", encoding="utf-8") as file:
#         queries = json.load(file)

#     summary_cases = []
#     batch_size = 2  

#     # **批处理 cases**
#     batch_inputs = []
#     for sample in cases:
#         batch_inputs.append([
#             [{"role": "system", "content": system_prompt}, {"role": "user", "content": sample["yishen_content"]}],
#             [{"role": "system", "content": system_prompt}, {"role": "user", "content": sample["ershen_content"]}]
#         ])
    
#     # **分批推理**
#     for i in tqdm(range(0, len(batch_inputs), batch_size)):
#         batch = batch_inputs[i: i + batch_size]
#         batch1 = [x[0] for x in batch]
#         batch2 = [x[1] for x in batch]
#         responses1 = batch_SummaryChat(batch1, tokenizer, model)
#         responses2 = batch_SummaryChat(batch2, tokenizer, model)
        
#         for r1, r2 in zip(responses1, responses2):
#             summary_cases.append({"yishen": r1, "ershen": r2})
        
#         torch.cuda.empty_cache()  

#     summary_queries = []
#     batch_inputs = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": sample["yishen_content"]}] for sample in queries]

#     # **分批推理 queries**
#     for i in tqdm(range(0, len(batch_inputs), batch_size)):
#         batch = batch_inputs[i: i + batch_size]
#         responses = batch_SummaryChat(batch, tokenizer, model)
        
#         for r in responses:
#             summary_queries.append({"yishen": r})
        
#         torch.cuda.empty_cache()

#     with open("./dataset/summary_cases.json", "w", encoding="utf-8") as file:
#         json.dump(summary_cases, file, indent=4, ensure_ascii=False)
#     with open("./dataset/summary_queries.json", "w", encoding="utf-8") as file:
#         json.dump(summary_queries, file, indent=4, ensure_ascii=False)
