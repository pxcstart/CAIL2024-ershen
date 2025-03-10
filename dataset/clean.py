import json
import re

with open("./summary_cases.json", "r", encoding="utf-8") as file:
    cases = json.load(file)
with open("./summary_queries.json", "r", encoding="utf-8") as file:
    queries = json.load(file)

cleaned_cases = []
cleaned_queries = []
for case in cases:
    new_case = {"yishen":{}, "ershen":{}}
    parts = case["yishen"].split("<法院判决结果>：")
    if len(parts)==2:
        background = parts[0].replace("<案件基本信息>：", "").strip()
        fact = parts[1].strip()
        new_case["yishen"] = {"background":background, "fact":fact}
    else:
        parts = case["yishen"].split("法院判决结果是：")
        if len(parts)==2:
            background = parts[0].replace("<案件基本信息>：", "").strip()
            fact = parts[1].strip()
            new_case["yishen"] = {"background":background, "fact":fact}
        else:
            print(case["yishen"])
            continue
    parts = case["ershen"].split("<法院判决结果>：")
    if len(parts)==2:
        background = parts[0].replace("<案件基本信息>：", "").strip()
        fact = parts[1].strip()
        new_case["ershen"] = {"background":background, "fact":fact}
    else:
        parts = case["ershen"].split("法院判决结果是：")
        if len(parts)==2:
            background = parts[0].replace("<案件基本信息>：", "").strip()
            fact = parts[1].strip()
            new_case["ershen"] = {"background":background, "fact":fact}
        else:
            parts = case["yishen"].split("法院判决结果是：")
            if len(parts)==2:
                background = parts[0].replace("<案件基本信息>：", "").strip()
                fact = parts[1].strip()
                new_case["yishen"] = {"background":background, "fact":fact}
            else:
                print(case["yishen"])
                continue
    cleaned_cases.append(new_case)

for query in queries:
    new_query = {}
    parts = query["yishen"].split("<法院判决结果>：")
    if len(parts)==2:
        background = parts[0].replace("<案件基本信息>：", "").strip()
        fact = parts[1].strip()
        new_query["yishen"] = {"background":background, "fact":fact}
    else:
        continue
    cleaned_queries.append(new_query)

with open("./cleaned_cases.json", "w", encoding="utf-8") as file:
    json.dump(cleaned_cases, file, indent=4, ensure_ascii=False)
with open("./cleaned_queries.json", "w", encoding="utf-8") as file:
    json.dump(cleaned_queries, file, indent=4, ensure_ascii=False)
