from bleurt import score
from rouge_score import rouge_scorer
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import evaluate

def BLEURT_score(predictions, references):
    bleurt_ops = score.create_bleurt_ops()
    scores = []
    for ref, pred in tqdm(zip(references, predictions), total=len(references), desc="Computing BLEURT scores"):
        ref_tensor = tf.constant([ref])
        pred_tensor = tf.constant([pred])
        bleurt_out = bleurt_ops(references=ref_tensor, candidates=pred_tensor)
        scores.append(bleurt_out["predictions"][0])
    return {"BLEU_mean": np.mean(scores), "BLEU_std": np.std(scores)}

def BERT_score(predictions, references):
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        rescale_with_baseline=True,
    )
    precision = results["precision"]
    recall = results["recall"]
    f1 = results["f1"]
    return {"BERT_f1_mean": np.mean(f1), "BERT_f1_std": np.std(f1)}

def Rouge_score(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = []
    for ref,pred in tqdm(zip(references, predictions), total=len(references), desc="Computing Rouge scores"):
        score = scorer.score(ref, pred)
        scores.append(score['rougeL'])
    return {"Rouge_L_mean": np.mean(scores), "Rouge_L_std": np.std(scores)}

    