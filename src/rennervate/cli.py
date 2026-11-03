"""Rennervate command-line interface."""

import argparse
import math
import sys
from collections.abc import Sequence

from .models import MODELS, get_model
from .runtime import LocalModel, Options, RennervateError, download


def _positive(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Enter a positive whole number.") from exc
    if number < 1 or number >= 8192:
        raise argparse.ArgumentTypeError("Enter a whole number between 1 and 8191.")
    return number


def _temperature(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Enter a number between 0 and 2.") from exc
    if not math.isfinite(number) or not 0 <= number <= 2:
        raise argparse.ArgumentTypeError("Enter a number between 0 and 2.")
    return number


def _model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=[model.key for model in MODELS],
        default="4b",
        help="Model to use (default: 4b).",
    )
    parser.add_argument("--cache-dir", help="Folder for downloaded model files.")


def _chat_options(parser: argparse.ArgumentParser) -> None:
    _model_options(parser)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Where to run the model (default: auto).",
    )
    parser.add_argument("--compact", action="store_true", help="Use less NVIDIA graphics memory.")
    parser.add_argument("--offline", action="store_true", help="Use downloaded files only.")
    parser.add_argument("--max-tokens", type=_positive, default=512, help="Answer length limit.")
    parser.add_argument(
        "--temperature",
        type=_temperature,
        default=0.7,
        help="Response variety from 0 to 2 (default: 0.7).",
    )
    parser.add_argument("--system", help="Instructions to use throughout the conversation.")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="rennervate",
        description="Chat with the Rennervate model family.",
        epilog="Run 'rennervate' to start a conversation. Contact: team@banzhuan.ai.",
    )
    commands = root.add_subparsers(dest="command")
    _chat_options(commands.add_parser("chat", help="Start a conversation."))
    single = commands.add_parser("run", help="Ask a single question.")
    _chat_options(single)
    single.add_argument("prompt", help="Your question, or - to read standard input.")
    commands.add_parser("models", help="List available models.")
    _model_options(commands.add_parser("download", help="Download a model for later use."))
    return root


def _safe_text(text: str) -> str:
    return "".join(
        char for char in text if char in "\n\t" or (ord(char) >= 32 and not 127 <= ord(char) <= 159)
    )


def _answer(engine: LocalModel, messages: list[dict[str, str]]) -> str:
    chunks = []
    stream = engine.stream(messages)
    try:
        for chunk in stream:
            chunks.append(chunk)
            print(_safe_text(chunk), end="", flush=True)
    finally:
        stream.close()
    print()
    return "".join(chunks)


def _chat(engine: LocalModel, system: str | None) -> int:
    print(f"{engine.model_info.name}\nType /help for help, /clear to start fresh, /exit to leave.")
    history = [{"role": "system", "content": system}] if system else []
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except EOFError:
            print()
            return 0
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            return 0
        if prompt == "/help":
            print("/clear starts a new conversation. /exit leaves. Ctrl+C also exits.")
            continue
        if prompt == "/clear":
            history = [{"role": "system", "content": system}] if system else []
            print("Conversation cleared.")
            continue
        if prompt.startswith("/"):
            print("Unknown command. Type /help for available commands.")
            continue
        current = [*history, {"role": "user", "content": prompt}]
        print("Rennervate: ", end="", flush=True)
        answer = _answer(engine, current)
        history = [*current, {"role": "assistant", "content": answer}]


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        arguments = ["chat"]
    args = parser().parse_args(arguments)
    if args.command == "models":
        for model in MODELS:
            print(f"{model.key}  {model.name}\n    https://huggingface.co/{model.repository}")
        return 0
    try:
        model = get_model(args.model)
        if args.command == "download":
            print(f"Downloading {model.name}...", file=sys.stderr)
            location = download(model, Options(cache_dir=args.cache_dir))
            print(location)
            return 0
        options = Options(
            device=args.device,
            compact=args.compact,
            offline=args.offline,
            cache_dir=args.cache_dir,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        engine = LocalModel(model, options)
        if args.command == "chat":
            if not sys.stdin.isatty():
                raise RennervateError("For piped input, use 'rennervate run -'.")
            return _chat(engine, args.system)
        prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
        if not prompt.strip():
            raise RennervateError("Enter a non-empty prompt.")
        history = [{"role": "system", "content": args.system}] if args.system else []
        history.append({"role": "user", "content": prompt})
        _answer(engine, history)
        return 0
    except (RennervateError, ValueError) as exc:
        print(f"Rennervate: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nConversation ended.", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return 0
