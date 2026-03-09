import gradio as gr

from src.analyzer import SentimentAnalyzer
from src.vader_analyzer import VaderAnalyzer


def create_app():
    transformer_analyzer = SentimentAnalyzer()
    vader_analyzer = VaderAnalyzer()

    def analyze(text: str, engine: str):
        if not text.strip():
            return "Please enter some text."

        analyzer = transformer_analyzer if engine == "Transformer" else vader_analyzer
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

        output_parts = []
        for line in lines:
            result = analyzer.analyze(line)
            scores_str = " | ".join(
                f"{k}: {v:.2%}" for k, v in result["scores"].items()
            )
            output_parts.append(
                f"Text: {result['text']}\n"
                f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2%})\n"
                f"Scores: {scores_str}\n"
            )
        return "\n".join(output_parts)

    demo = gr.Interface(
        fn=analyze,
        inputs=[
            gr.Textbox(
                lines=5,
                placeholder="Enter text to analyze (one per line for batch mode)...",
                label="Input Text",
            ),
            gr.Radio(
                choices=["Transformer", "VADER"],
                value="Transformer",
                label="Analysis Engine",
            ),
        ],
        outputs=gr.Textbox(label="Results", lines=10),
        title="Sentiment Analysis Tool",
        description="Analyze the sentiment of text as Positive, Negative, or Neutral. "
        "Transformer engine uses a RoBERTa model; VADER is a lightweight rule-based analyzer.",
        examples=[
            ["I absolutely love this product! Best purchase ever!", "Transformer"],
            ["This is the worst experience I've ever had.", "VADER"],
            ["The weather is okay today.\nI hate Mondays.\nThis cake is delicious!", "Transformer"],
        ],
    )
    return demo


def launch_app():
    demo = create_app()
    demo.launch()
