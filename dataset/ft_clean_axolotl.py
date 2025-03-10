import json

with open("./trainset.json", "r", encoding="utf-8") as file:
    trainset = json.load(file)
alpaca_set = []
for sample in trainset:
    item = {}
    item["instruction"]=sample[0]["content"].replace('##指令：','')
    item["input"]=sample[1]["content"].replace('##输入：','').replace("##输出：",'')
    item["output"]=sample[2]["content"]
    alpaca_set.append(item)
with open("./alpaca_trainset.jsonl", "w", encoding="utf-8") as f:
    for item in alpaca_set:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


