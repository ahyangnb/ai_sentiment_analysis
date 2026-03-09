import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)


class VaderAnalyzer:
    def __init__(self):
        self._analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> dict:
        scores = self._analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": round(abs(compound), 4),
            "scores": {
                "Positive": round(scores["pos"], 4),
                "Neutral": round(scores["neu"], 4),
                "Negative": round(scores["neg"], 4),
            },
        }

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        return [self.analyze(t) for t in texts]
