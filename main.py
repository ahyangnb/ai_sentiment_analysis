import argparse
import sys


def print_result(result: dict):
    scores_str = " | ".join(f"{k}: {v:.2%}" for k, v in result["scores"].items())
    print(f"  Text:       {result['text']}")
    print(f"  Sentiment:  {result['sentiment']} ({result['confidence']:.2%})")
    print(f"  Scores:     {scores_str}")
    print()


def get_analyzer(engine: str):
    if engine == "transformer":
        from src.analyzer import SentimentAnalyzer
        return SentimentAnalyzer()
    else:
        from src.vader_analyzer import VaderAnalyzer
        return VaderAnalyzer()


def interactive_mode(analyzer):
    print("Sentiment Analysis - Interactive Mode (type 'quit' to exit)")
    print("-" * 50)
    while True:
        try:
            text = input("\nEnter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if text.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not text:
            continue
        result = analyzer.analyze(text)
        print_result(result)


def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis Tool")
    parser.add_argument("--text", "-t", type=str, help="Text to analyze")
    parser.add_argument("--file", "-f", type=str, help="File with texts (one per line)")
    parser.add_argument(
        "--engine", "-e",
        choices=["transformer", "vader"],
        default="transformer",
        help="Analysis engine (default: transformer)",
    )
    parser.add_argument("--web", "-w", action="store_true", help="Launch web UI")
    args = parser.parse_args()

    if args.web:
        from src.app import launch_app
        launch_app()
        return

    analyzer = get_analyzer(args.engine)

    if args.text:
        result = analyzer.analyze(args.text)
        print_result(result)
    elif args.file:
        with open(args.file) as f:
            texts = [line.strip() for line in f if line.strip()]
        for result in analyzer.analyze_batch(texts):
            print_result(result)
    else:
        interactive_mode(analyzer)


if __name__ == "__main__":
    main()
