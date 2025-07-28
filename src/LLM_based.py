from ollama import chat , Client
import pandas as pd
from calculate_uncertainty import compute_uncertainty
from tqdm import tqdm

client = Client()

def ask_llm(prompt: str,
                 model: str = "llama3.2:1b",
                 temperature: float = 0.8) -> str:
    res = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return res["message"]["content"].strip().lower()

def binary_prob_mc(prompt: str,
                   model: str = "llama3.2:1b",
                   N: int = 5,
                   temperature: float = 0.8) -> float:
    """Run the prompt N times, each must answer yes/no, return P(yes)."""
    yes = 0
    for _ in range(N):
        out = ask_llm(prompt, model, temperature)
        yes += out.startswith("y")          # treat any yes-variant as yes
    return yes / N                          # probability in [0,1]


def run_self_questioning(df_split: pd.DataFrame,
                         docs: dict,
                         N_mc: int = 5,
                         temp: float = 0.8,
                         model: str = "llama3.2:1b") -> pd.DataFrame:

    records, out_of = [], len(df_split)
    # for idx, row in enumerate(df_split.itertuples(), 1):
    for idx, row in enumerate(tqdm(df_split.itertuples(), total=len(df_split)), 1):

        # print(row)
        bbcid   = str(row.bbcid)
        summ    = row.summary
        doc     = docs[bbcid]

        q1 = ("Q1: Is all information in this summary accurate given the source? "
              "Only answer with 'yes' or 'no'. "
              f"Source: {doc} Summary: {summ}")
        q2 = ("Q2: Does the summary contain any information that conflicts with the source? "
              "Only answer with 'yes' or 'no'. "
              f"Source: {doc} Summary: {summ}")

        # Monte-Carlo probabilities
        p_true    = binary_prob_mc(q1, model, N_mc, temp)
        p_contrad = binary_prob_mc(q2, model, N_mc, temp)

        # Majority yes/no
        ans1 = "yes" if p_true    >= 0.5 else "no"
        ans2 = "yes" if p_contrad >= 0.5 else "no"

        verdict = ("Entailed"      if ans1 == "yes" and ans2 == "no" else
                   "Contradicted"  if ans1 == "no"  and ans2 == "yes" else
                   "Unsupported")

        unc = compute_uncertainty(doc, summ)          # inverse perplexity

        records.append({
            "bbcid":          bbcid,
            "summary":        summ,
            "system":         row.system,
            # raw Monte-Carlo probs
            "p_true":         p_true,        # CEHD → P(True)
            "p_contrad":      p_contrad,     # CEHD → P(InputContradict)
            # majority answers (optional)
            "ans_factual":    ans1,
            "ans_contradict": ans2,
            "self_verdict":   verdict,
            # other single-generation score
            "inv_ppl":        unc
        })

        print(f"{idx}/{out_of}", end="\r")

    return pd.DataFrame(records)



def prompt_paring_prob(doc, summ, N_mc: int = 5, temp: float = 0.8, model: str = "llama3.2:1b"):
    q1 = ("Q1: Is all information in this summary accurate given the source? "
          "Only answer with 'yes' or 'no'. "
          f"Source: {doc} Summary: {summ}")
    q2 = ("Q2: Does the summary contain any information that conflicts with the source? "
          "Only answer with 'yes' or 'no'. "
          f"Source: {doc} Summary: {summ}")
    p_true = binary_prob_mc(q1, model, N_mc, temp)
    p_contrad = binary_prob_mc(q2, model, N_mc, temp)

    return p_true , p_contrad
# def run_self_questioning(df_split , docs):
#     records = []
#     out_of = len(df_split)
#     counter = 0
#     for _, row in df_split.iterrows():
#         bbcid   = str(row["bbcid"])
#         summ    = row["summary"]
#         doc     = docs[bbcid]
#
#         q1 = (
#             "Q1: Is all information in this summary accurate given the source? "
#             "Only answer with 'yes' or 'no'. "
#             f"Source: {doc} Summary: {summ}"
#         )
#         q2 = (
#             "Q2: Does the summary contain any information that conflicts with the source? "
#             "Only answer with 'yes' or 'no'. "
#             f"Source: {doc} Summary: {summ}"
#         )
#
#         ans1 = ask_llm(q1)
#         ans2 = ask_llm(q2)
#
#         if "yes" in ans1 and "no" in ans2:
#             verdict = "Entailed"
#         elif "no" in ans1 and "yes" in ans2:
#             verdict = "Contradicted"
#         else:
#             verdict = "Unsupported"
#
#         unc = compute_uncertainty(doc, summ)
#
#         records.append({
#             "bbcid":         bbcid,
#             "summary": summ ,
#             "system":        row["system"],
#             "ans_factual":   ans1,
#             "ans_contradict":ans2,
#             "self_verdict":  verdict,
#             'uncertainty':  unc
#         })
#         counter += 1
#         print(f"{counter} out of {out_of}")
#     return pd.DataFrame(records)
#
#
