from transformers import pipeline
from gensim.summarization import summarize as gensim_summarize
from rouge_score import rouge_scorer

_summarizer_pipeline = None

def _get_summarizer():
    global _summarizer_pipeline
    if _summarizer_pipeline is None:
        _summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    return _summarizer_pipeline

def generate_summary(text, min_length=30, max_length=130):
    try:
        summarizer = _get_summarizer()
        out = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return out[0]["summary_text"]
    except Exception as e:
        return "Error in summarization: " + str(e)

def textrank_summary(text):
    try:
        s = gensim_summarize(text, ratio=0.2)
        if not s.strip():
            return "Text too short or not suitable for TextRank summarization."
        return s
    except Exception:
        return "Text too short or not suitable for TextRank summarization."

def compare_summaries(original, summary1, summary2):
    scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
    s1 = scorer.score(original, summary1)
    s2 = scorer.score(original, summary2)
    out = f"ROUGE scores:\\nSummary1 (rouge1): {s1['rouge1'].fmeasure:.3f}, Summary1 (rougeL): {s1['rougeL'].fmeasure:.3f}\\n"
    out += f"Summary2 (rouge1): {s2['rouge1'].fmeasure:.3f}, Summary2 (rougeL): {s2['rougeL'].fmeasure:.3f}\\n"
    return out
