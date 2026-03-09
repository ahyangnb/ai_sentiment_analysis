from transformers import pipeline


class SentimentAnalyzer:
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    LABEL_MAP = {
        "negative": "Negative",
        "neutral": "Neutral",
        "positive": "Positive",
    }

    def __init__(self):
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            print(f"Loading model: {self.MODEL_NAME} (first run will download ~500MB)...")
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.MODEL_NAME,
                top_k=None,
            )
        return self._pipeline

    def analyze(self, text: str) -> dict:
        pipe = self._load()
        results = pipe(text)[0]
        scores = {self.LABEL_MAP[r["label"]]: round(r["score"], 4) for r in results}
        best = max(results, key=lambda r: r["score"])
        return {
            "text": text,
            "sentiment": self.LABEL_MAP[best["label"]],
            "confidence": round(best["score"], 4),
            "scores": scores,
        }

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        return [self.analyze(t) for t in texts]
