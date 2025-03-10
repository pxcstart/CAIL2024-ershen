from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import json

def minibatch(batch_size, *tensors, **kwargs):
    batch_size = kwargs.get('batch_size', batch_size)
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

if __name__=="__main__":
    model = SentenceTransformer("../model/m3e-base")
    with open("./dataset/cleaned_cases.json", "r", encoding="utf-8") as file:
        cases = json.load(file)
    case_embs = []
    for samples in minibatch(8, cases):
        background = [sample['yishen']["background"] for sample in samples]
        embs = model.encode(background)
        case_embs.append(embs)
    case_embs = np.concatenate(case_embs, axis=0).reshape(-1, 768)
    case_embs = torch.tensor(case_embs).to("cuda")
    torch.save(case_embs,"./dataset/case_embs.pt")
