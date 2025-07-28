from src.inference import infer
import gradio as gr
import pandas as pd


KEY_LABELS = {
    "raw_p_true":         "LLM P(True)",
    "raw_p_contrad":      "LLM P(Contra.)",
    "raw_inv_ppl":        "1 / PPL",
    "raw_fact_score":     "FactScore",
    "raw_entity_precision":"Entity Prec.",
    "raw_triple_precision":"Triple Prec.",
    "raw_sem_entail":     "Sem. Entail.",
    "raw_topic_drift":    "Topic Drift",
    "p_true_cal":         "P(True) (Calibrated)",
    "p_contrad_cal":      "P(Contra.) (Calibrated)",
    "inv_ppl_cal":        "1/PPL (Calibrated)",
    "fact_score_cal":     "FactScore (Calibrated)",
    "entity_precision_cal":"Entity Prec. (Calibrated)",
    "triple_precision_cal":"Triple Prec. (Calibrated)",
    "sem_entail_cal":     "Sem. Entail. (Calibrated)",
    "topic_drift_cal":    "Topic Drift (Calibrated)"
}


def run_inference(article: str, summary: str):
    out = infer(article, summary)

    pred_str = "✅ Faithful" if out["predicted"] == 1 else "❌ Hallucinated"

    prob = out["probability"]

    rows = []
    for k in sorted(out):
        if k.startswith("raw_") or k.endswith("_cal"):
            label = KEY_LABELS.get(k, k)    # fallback to key if not found
            rows.append([label, out[k]])
    df = pd.DataFrame(rows, columns=["Feature", "Value"])

    return pred_str, prob, df

title = "📢 Hallucination Detection Demo"
description = """
Paste in a full **article** on the left, and its **summary** on the right.  
Click **Detect** to see:
1. Whether the summary is **faithful** or **hallucinated**  
2. The model’s confidence (probability)  
3. All intermediate raw & calibrated feature scores  
"""

demo = gr.Interface(
    fn=run_inference,
    inputs=[
        gr.Textbox(lines=8, label="Article"),
        gr.Textbox(lines=4, label="Summary"),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Number(label="Probability of Faithfulness", precision=3),
        gr.Dataframe(
            headers=["Feature","Value"],
            datatype=["str","str"],
            label="Raw & Calibrated Features"
        )
    ],
    title=title,
    description=description,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
