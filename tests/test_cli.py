import io
import subprocess
import sys

import pytest

from rennervate import cli
from rennervate.runtime import RennervateError


class Terminal(io.StringIO):
    def isatty(self):
        return True


@pytest.fixture
def engine(monkeypatch):
    class FakeModel:
        calls = []
        instances = []
        closed = False

        def __init__(self, model, options):
            self.model_info = model
            self.options = options
            self.instances.append(self)

        def stream(self, messages):
            self.calls.append([dict(message) for message in messages])
            try:
                yield "Hello "
                yield "world."
            finally:
                self.closed = True

    monkeypatch.setattr(cli, "LocalModel", FakeModel)
    return FakeModel


def test_model_listing_does_not_load_runtime(capsys, engine):
    assert cli.main(["models"]) == 0
    output = capsys.readouterr().out
    assert "4b" in output and "8b" in output
    assert len(engine.instances) == 0


def test_single_answer_and_selected_model(capsys, engine):
    assert cli.main(["run", "--model", "8b", "--system", "Be brief.", "Hi"]) == 0
    assert capsys.readouterr().out == "Hello world.\n"
    assert engine.instances[0].model_info.key == "8b"
    assert engine.calls == [
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ]
    ]
    assert engine.instances[0].closed


def test_piped_prompt(monkeypatch, engine):
    monkeypatch.setattr(sys, "stdin", io.StringIO("A multiline\nprompt"))
    assert cli.main(["run", "-"]) == 0
    assert engine.calls[0][-1]["content"] == "A multiline\nprompt"


def test_chat_retains_and_clears_history(monkeypatch, engine, capsys):
    monkeypatch.setattr(
        sys, "stdin", Terminal("\n/help\n/unknown\nHi\nAgain\n/clear\nNew\n/exit\n")
    )
    assert cli.main(["chat", "--system", "Brief"]) == 0
    assert [len(call) for call in engine.calls] == [2, 4, 2]
    assert engine.calls[1][-2] == {"role": "assistant", "content": "Hello world."}
    assert engine.calls[2][0] == {"role": "system", "content": "Brief"}
    assert "Conversation cleared." in capsys.readouterr().out


def test_no_arguments_starts_chat_without_loading_on_exit(monkeypatch, engine):
    monkeypatch.setattr(sys, "stdin", Terminal("/exit\n"))
    assert cli.main([]) == 0
    assert engine.calls == []


def test_chat_eof(monkeypatch, engine):
    monkeypatch.setattr(sys, "stdin", Terminal(""))
    assert cli.main([]) == 0


@pytest.mark.parametrize("arguments", [["run", " "], ["chat"]])
def test_bad_input_does_not_load_model(arguments, monkeypatch, engine, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert cli.main(arguments) == 1
    assert engine.calls == []
    assert "Rennervate:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--max-tokens", "0"),
        ("--max-tokens", "8192"),
        ("--max-tokens", "abc"),
        ("--temperature", "nan"),
        ("--temperature", "inf"),
        ("--temperature", "3"),
        ("--model", "unknown"),
    ],
)
def test_invalid_arguments(flag, value, engine):
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", flag, value, "Hello"])
    assert exc.value.code == 2
    assert engine.calls == []


def test_compact_cpu_rejected_before_loading(engine, capsys):
    assert cli.main(["run", "--compact", "--device", "cpu", "Hi"]) == 1
    assert "NVIDIA" in capsys.readouterr().err
    assert engine.instances == []


def test_download_routes_model_and_cache(monkeypatch, capsys):
    calls = []

    def fake_download(model, options):
        calls.append((model, options))
        return "/models/example"

    monkeypatch.setattr(cli, "download", fake_download)
    assert cli.main(["download", "--model", "8b", "--cache-dir", "/models"]) == 0
    assert calls[0][0].key == "8b"
    assert calls[0][1].cache_dir == "/models"
    output = capsys.readouterr()
    assert output.out == "/models/example\n"
    assert "Downloading" in output.err


def test_generation_failure_is_reported(monkeypatch, engine, capsys):
    def failing(self, messages):
        yield "Partial "
        raise RennervateError("Try again.")

    monkeypatch.setattr(engine, "stream", failing)
    assert cli.main(["run", "Hi"]) == 1
    assert "Try again." in capsys.readouterr().err


def test_interrupt_closes_stream(monkeypatch, engine, capsys):
    def interrupting(self, messages):
        try:
            raise KeyboardInterrupt
            yield
        finally:
            self.closed = True

    monkeypatch.setattr(engine, "stream", interrupting)
    assert cli.main(["run", "Hi"]) == 130
    assert engine.instances[0].closed
    assert "ended" in capsys.readouterr().err


def test_terminal_control_characters_are_removed():
    assert cli._safe_text("a\x1bb\x07c\x9bd\n\t世界") == "abcd\n\t世界"


def test_module_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "rennervate", "models"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Rennervate 4B Instruct" in result.stdout
