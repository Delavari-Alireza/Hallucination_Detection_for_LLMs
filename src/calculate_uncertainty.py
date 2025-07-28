import math, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


UNCERT_MODEL = "google/flan-t5-small"
tok_unc  = AutoTokenizer.from_pretrained(UNCERT_MODEL)
model_unc= AutoModelForSeq2SeqLM.from_pretrained(
    UNCERT_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

# make sure we have a pad token
if tok_unc.pad_token_id is None:
    tok_unc.pad_token_id = tok_unc.eos_token_id

#Compute inverse-perplexity
def compute_uncertainty(source: str, summary: str) -> float:
    """
    Returns 1 / PPL(summary | source) using flan-t5-small.
    """
    # Encode source for the encoder
    enc = tok_unc("summarize: " + source, return_tensors="pt").to(model_unc.device)
    # Encode summary as labels
    with tok_unc.as_target_tokenizer():
        labels = tok_unc(summary, return_tensors="pt").input_ids.to(model_unc.device)

    # Forward with labels gives cross-entropy
    with torch.no_grad():
        out = model_unc(**enc, labels=labels)
        loss = out.loss.item()       # mean CE per token

    ppl = math.exp(loss)            # perplexity
    return 1.0 / ppl                # inverse, in (0,1]

# if __name__=="__main__":
#     src  = "The Eiffel Tower is in Paris and opened in 1889."
#     summ = "The Eiffel Tower opened in 1889 in Paris."
#     unc  = compute_uncertainty_t5(src, summ)
#     print(f"Uncertainty (1/PPL) = {unc:.4f}")
