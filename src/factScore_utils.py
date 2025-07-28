from sentence_transformers import SentenceTransformer, util
import spacy
import json
import pandas as pd

nlp = spacy.load("en_core_web_sm")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_entities(text: str):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def match_ent(e_summ: str, e_src: str, thresh=0.8) -> bool:
    if e_summ == e_src:
        return True
    emb1 = embedder.encode(e_summ, convert_to_tensor=True)
    emb2 = embedder.encode(e_src, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item() >= thresh

def match_verb(v_summ: str, v_src: str, thresh=0.8) -> bool:
    if v_summ == v_src:
        return True
    emb1 = embedder.encode(v_summ, convert_to_tensor=True)
    emb2 = embedder.encode(v_src, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item() >= thresh

def extract_svo_triples(text: str):
    """
    - Sentence‐level SVO
    - Verbs: ROOT, ccomp, conj (VERB or AUX)
    - Subjects: nsubj/nsubjpass (drop pronouns)
    - Objects: dobj/attr/dative + first pobj under prep + ccomp handled separately
    - Expand every object via its subtree
    - Drop objects whose text starts with "their "
    - Dedupe via a set
    """
    doc = nlp(text)
    triples = set()

    for sent in doc.sents:
        # Collect candidate verbs
        verbs = [t for t in sent
                 if t.pos_ in ("VERB","AUX") and t.dep_ in ("ROOT","ccomp","conj")]
        # plus conjuncts of those
        for v in list(verbs):
            verbs.extend([c for c in v.conjuncts if c.pos_ in ("VERB","AUX")])

        for verb in verbs:
            # Subjects
            subs = [c for c in verb.children
                    if c.dep_ in ("nsubj","nsubjpass") and c.pos_!="PRON"]
            # inherit for conj w/o own subs
            if not subs and verb.dep_=="conj":
                subs = [c for c in verb.head.children
                        if c.dep_ in ("nsubj","nsubjpass") and c.pos_!="PRON"]
            if not subs:
                continue

            # Objects
            objs = []
            for c in verb.children:
                if c.dep_ in ("dobj","attr","dative"):
                    objs.append(c)
                if c.dep_ == "prep":
                    # first pobj under this prep
                    for pobj in c.children:
                        if pobj.dep_ == "pobj":
                            objs.append(pobj)
                            break
                # handle clausal complements as separate SVOs
                if c.dep_ == "ccomp":
                    nested = extract_svo_triples(c.text)
                    for (ns,nv,no) in nested:
                        for sub in subs:
                            triples.add((sub.text, verb.lemma_, f"{nv} {no}"))

            if not objs:
                continue

            # Build and filter triples
            for sub in subs:
                subj_text = sub.text
                for obj in objs:
                    if obj.pos_ in ("PRON","NUM"):
                        continue
                    # full subtree span
                    toks = list(obj.subtree)
                    span = sent[toks[0].i - sent.start : toks[-1].i - sent.start + 1]
                    obj_text = span.text
                    if obj_text.lower().startswith("their "):
                        continue
                    triples.add((subj_text, verb.lemma_, obj_text))

    return list(triples)


def compute_fact_score(summary: str, source: str, alpha: float = 0.5,
                       ent_thresh: float = 0.8, verb_thresh: float = 0.8):
    # Extract
    summ_ents   = extract_entities(summary)
    summ_svos   = extract_svo_triples(summary)
    source_svos = extract_svo_triples(source)

    # Entity precision
    unsupported_entities = []
    sup_ents = 0
    for ent in summ_ents:
        # does this entity match any subject or object in source_svos?
        found = any(
            match_ent(ent, subj, ent_thresh) or match_ent(ent, obj, ent_thresh)
            for subj, _, obj in source_svos
        )
        if found:
            sup_ents += 1
        else:
            unsupported_entities.append(ent)
    ent_prec = sup_ents / len(summ_ents) if summ_ents else 1.0

    # Triple precision
    unsupported_triples = []
    sup_trips = 0
    for s_s, v_s, o_s in summ_svos:
        # look for ANY source triple that matches all three components
        match_found = any(
            match_ent(s_s, s_src, ent_thresh) and
            match_ent(o_s, o_src, ent_thresh) and
            match_verb(v_s, v_src, verb_thresh)
            for s_src, v_src, o_src in source_svos
        )
        if match_found:
            sup_trips += 1
        else:
            unsupported_triples.append((s_s, v_s, o_s))
    tri_prec = sup_trips / len(summ_svos) if summ_svos else 0

    # Combined FactScore (if you need it)
    fact_score = alpha * ent_prec + (1 - alpha) * tri_prec

    return {
        "fact_score":           fact_score,
        "entity_precision":     ent_prec,
        "triple_precision":     tri_prec,
        "unsupported_entities": unsupported_entities,
        "unsupported_triples":  unsupported_triples
    }