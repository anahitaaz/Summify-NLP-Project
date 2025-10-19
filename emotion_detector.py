from transformers import pipeline

_emotion_pipeline = None

def _get_emotion_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is None:
        _emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)
    return _emotion_pipeline

def detect_emotions(text, return_all=False):
    pipe = _get_emotion_pipeline()
    try:
        scores = pipe(text)[0]
        sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
        if return_all:
            return [(d['label'], d['score']) for d in sorted_scores]
        top = sorted_scores[0]
        return top['label'], top['score']
    except Exception:
        if return_all:
            return [("neutral", 1.0)]
        return "neutral", 1.0
