import spacy
import torch
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer, util

nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
topic_model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

def compute_semantic_entailment(source: str, summary: str) -> float:
    """
    Returns ScoreNLI = max_{s∈src_sents, h∈sum_sents}( P_entailment - P_contradiction )
    using cross-encoder/nli-deberta-v3-base.
    Scores range in [-1,1]: +1 = strong entailment, -1 = strong contradiction.
    """
    # 1. split into sentences
    src_sents  = [sent.text for sent in nlp(source).sents]
    sum_sents  = [sent.text for sent in nlp(summary).sents]

    best_score = -1.0  # worst-case
    for prem in src_sents:
        for hyp in sum_sents:
            # get logits for [contradiction, neutral, entailment]
            logits = nli_model.predict([(prem, hyp)])[0]
            probs  = torch.softmax(torch.tensor(logits), dim=0).numpy()
            p_contradict, p_neutral, p_entail = probs
            score = p_entail - p_contradict
            if score > best_score:
                best_score = score

    return best_score


def compute_topic_drift(source: str, summary: str) -> float:
    """
    Returns TopicDrift = 1 - cosine( embed(source), embed(summary) )
    where embeddings come from all-MiniLM-L6-v2.
    Drift in [0,2], but practically in [0,1].
    """
    emb_src = topic_model.encode(source,  convert_to_tensor=True)
    emb_sum = topic_model.encode(summary, convert_to_tensor=True)
    cos_sim = util.cos_sim(emb_src, emb_sum).item()
    drift   = 1.0 - cos_sim
    return drift

#
# if __name__ == "__main__":
#     example_src = (
#         "The visitors led briefly through Vasil Lobzhanidze's early try, "
#         "but Scotland raced ahead ... and got their reward."
#     )
#     example_sum = "Scotland dominated after an early Georgian try and ran out convincing winners."
#
#     nli_score  = compute_semantic_entailment(example_src, example_sum)
#     drift_score = compute_topic_drift(example_src, example_sum)
#
#     print(f"Semantic Entailment Score: {nli_score:.3f}  (+1=entail, -1=contra)")
#     print(f"Topic Drift Score:         {drift_score:.3f}  (0=on-topic, 1=off-topic)")
#
#
