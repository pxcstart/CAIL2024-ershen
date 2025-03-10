import json

case_path = "/home/peter/Challenge/origin_data/cases.json"
with open(case_path, "r", encoding="utf-8") as file:
    data = json.load(file)
case_samples = []
case_id = 0
for sample in data:
    case_tmp = {}

    yishen_res = sample["yishen_content"].split("审理终结。\n\n")
    if len(yishen_res) == 1:
        yishen_res = sample["yishen_content"].split("参加诉讼。\n\n")
    if len(yishen_res) == 1:
        continue
    yishen_content = yishen_res[1]

    res = yishen_content.split("判决如下：")
    res2 = res[0].split("\n\n综上")
    if len(res2)==1: continue
    case_tmp["yishen_background"] = res2[0]
    if "驳回上诉，维持原判" in res[-1]: case_tmp["yishen_fact"]="驳回上诉，维持原判"
    else:
        yishen_fact = res[-1].split("\n\n如不服本判决")
        case_tmp["yishen_fact"] = yishen_fact[0].replace("\n\n", "")

    ershen_res = sample["ershen_content"].split("审理终结。\n\n")
    if len(ershen_res) == 1:
        ershen_res = sample["ershen_content"].split("参加诉讼。\n\n")
    if len(ershen_res) == 1:
        continue
    ershen_content = ershen_res[1]

    res = ershen_content.split("判决如下：")
    if "驳回上诉，维持原判" in res[-1]: case_tmp["ershen_fact"]="驳回上诉，维持原判"
    else:
        ershen_fact = res[-1].split("\n\n本判决为终审判决")
        case_tmp["ershen_fact"] = ershen_fact[0].replace("\n\n", "")

    case_tmp["id"] = case_id
    case_id += 1
    case_samples.append(case_tmp)
    
with open("/home/peter/Challenge/baseline/data/clean_cases.json", "w") as file:
    json.dump(case_samples, file, indent=4, ensure_ascii=False)


query_path = "/home/peter/Challenge/origin_data/queries.json"
with open(query_path, "r", encoding="utf-8") as file:
    data = json.load(file)
query_samples = []
query_id = 0
for sample in data:
    query_tmp = {}

    yishen_res = sample["yishen_content"].split("审理终结。\n\n")
    if len(yishen_res) == 1:
        yishen_res = sample["yishen_content"].split("参加诉讼。\n\n")
    if len(yishen_res) == 1:
        continue
    yishen_content = yishen_res[1]

    res = yishen_content.split("判决如下：")
    res2 = res[0].split("\n\n综上")
    if len(res2)==1: continue
    query_tmp["yishen_background"] = res2[0]
    if "驳回上诉，维持原判" in res[-1]: query_tmp["yishen_fact"]="驳回上诉，维持原判"
    else:
        yishen_fact = res[-1].split("\n\n如不服本判决")
        query_tmp["yishen_fact"] = yishen_fact[0].replace("\n\n", "")

    query_tmp["id"] = query_id
    query_id += 1
    query_samples.append(query_tmp)

with open("/home/peter/Challenge/baseline/data/clean_queries.json", "w") as file:
    json.dump(query_samples, file, indent=4, ensure_ascii=False)