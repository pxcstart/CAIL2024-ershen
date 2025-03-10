import json
import random
 
system_prompt = "##指令：假设你是一个精通各种法律知识的法官，下面我会给出案件的一审事实陈述和判决结果，现在该案件进入二审阶段，请你给出二审判决结果。\n"
system_content = {"role":"system", "content":system_prompt}
with open("./cleaned_cases.json", "r", encoding="utf-8") as file:
    samples = json.load(file)
train_size = int(len(samples)*0.8)
train_samples = random.sample(samples, train_size)
test_samples = [item for item in samples if item not in train_samples]
trainset = []
for sample in train_samples:
    user_input = f'##输入：1.一审事实陈述：{sample["yishen"]["background"]} \n 2.一审判决结果：{sample["yishen"]["fact"]} \n  ##输出：'
    user_content = {"role":"user", "content":user_input}
    assistant_content = {"role":"assistant", "content":f'{sample["ershen"]["fact"]}'}
    trainset.append([system_content, user_content, assistant_content])
testset = []
for sample in test_samples: 
    user_input = f'##输入：1.一审事实陈述：{sample["yishen"]["background"]} \n 2.一审判决结果：{sample["yishen"]["fact"]} \n  ##输出：'
    user_content = {"role":"user", "content":user_input}
    assistant_content = {"role":"assistant", "content":f'{sample["ershen"]["fact"]}'}
    testset.append([system_content, user_content, assistant_content])

with open(f"./trainset.json", "w", encoding="utf-8") as file:
    json.dump(trainset, file, indent=4, ensure_ascii=False)

with open(f"./testset.json", "w", encoding="utf-8") as file:
    json.dump(testset, file, indent=4, ensure_ascii=False)
