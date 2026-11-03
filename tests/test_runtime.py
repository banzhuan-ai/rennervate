import contextlib
import queue
import sys
import threading
from types import SimpleNamespace

import pytest

from rennervate.models import get_model
from rennervate.runtime import LocalModel, Options, RennervateError, download


def test_known_models_and_unknown_selection():
    assert get_model().key == "4b"
    assert get_model("8B").key == "8b"
    assert get_model(get_model("8b").repository).key == "8b"
    with pytest.raises(ValueError):
        get_model("somewhere/other-model")


@pytest.mark.parametrize(
    "options",
    [
        {"max_tokens": 0},
        {"max_tokens": 8192},
        {"temperature": float("nan")},
        {"temperature": -1},
        {"device": "other"},
        {"compact": True, "device": "cpu"},
    ],
)
def test_options_reject_invalid_values(options):
    with pytest.raises(ValueError):
        Options(**options)


def test_download_uses_selected_revision_and_only_model_files(monkeypatch):
    captured = {}

    def snapshot_download(repository, **kwargs):
        captured.update(repository=repository, **kwargs)
        return "/cache/model"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=snapshot_download,
        ),
    )
    assert download(get_model("8b"), Options(cache_dir="/cache", offline=True)) == "/cache/model"
    assert captured["repository"] == get_model("8b").repository
    assert captured["revision"] == get_model("8b").revision
    assert captured["local_files_only"]
    assert "*.safetensors" in captured["allow_patterns"]
    assert "*.py" not in captured["allow_patterns"]


class Inputs(dict):
    def __init__(self, length):
        super().__init__(input_ids=SimpleNamespace(shape=(1, length)))

    def to(self, device):
        return self


class Tokenizer:
    eos_token_id = 0

    def __init__(self):
        self.seen = []

    def apply_chat_template(self, messages, **kwargs):
        self.seen.append(([dict(m) for m in messages], kwargs))
        return " ".join(m["content"] for m in messages)

    def __call__(self, prompt, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return Inputs(len(prompt.split()))


def prepared_model(options=None):
    engine = LocalModel(get_model(), options or Options())
    engine.tokenizer = Tokenizer()
    engine.model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=8192), device="cpu"
    )
    return engine


def test_long_chat_drops_old_turns_and_keeps_system_and_latest_prompt():
    engine = prepared_model(Options(context_limit=12, max_tokens=4))
    messages = [
        {"role": "system", "content": "Keep this"},
        {"role": "user", "content": "An older user message"},
        {"role": "assistant", "content": "An older assistant message"},
        {"role": "user", "content": "The latest question"},
    ]
    engine._inputs(messages)
    assert engine.tokenizer.seen[-1][0] == [messages[0], messages[-1]]
    assert len(messages) == 4
    assert engine.tokenizer.seen[-1][1]["enable_thinking"] is False


def test_oversized_current_prompt_fails_without_silent_truncation():
    engine = prepared_model(Options(context_limit=8, max_tokens=4))
    with pytest.raises(RennervateError, match="too long"):
        engine._inputs([{"role": "user", "content": "one two three four five"}])


@pytest.fixture
def streaming_modules(monkeypatch):
    class Streamer:
        sentinel = object()

        def __init__(self, tokenizer, **kwargs):
            self.queue = queue.Queue()
            self.timeout = kwargs["timeout"]

        def __next__(self):
            item = self.queue.get(timeout=self.timeout)
            if item is self.sentinel:
                raise StopIteration
            return item

        def end(self):
            self.queue.put(self.sentinel)

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            inference_mode=contextlib.nullcontext,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            TextIteratorStreamer=Streamer,
            StoppingCriteria=object,
            StoppingCriteriaList=list,
        ),
    )


def test_streaming_returns_output_and_stops_worker(streaming_modules):
    engine = prepared_model()

    def generate(**kwargs):
        kwargs["streamer"].queue.put("Hello")
        kwargs["streamer"].queue.put(" world")
        kwargs["streamer"].end()

    engine.model.generate = generate
    assert list(engine.stream([{"role": "user", "content": "Hi"}])) == ["Hello", " world"]
    assert not any(t.name == "rennervate-generation" for t in threading.enumerate())


def test_worker_failure_is_propagated_without_hanging(streaming_modules):
    engine = prepared_model()

    def generate(**kwargs):
        raise RuntimeError("CUDA out of memory")

    engine.model.generate = generate
    with pytest.raises(RennervateError, match="not enough memory"):
        list(engine.stream([{"role": "user", "content": "Hi"}]))


def test_closing_stream_cancels_worker(streaming_modules):
    engine = prepared_model()
    completed = threading.Event()

    def generate(**kwargs):
        kwargs["streamer"].queue.put("Hello")
        while not kwargs["stopping_criteria"][0](None, None):
            completed.wait(0.01)
        completed.set()
        kwargs["streamer"].end()

    engine.model.generate = generate
    stream = engine.stream([{"role": "user", "content": "Hi"}])
    assert next(stream) == "Hello"
    stream.close()
    assert completed.is_set()


def test_greedy_generation_omits_sampling_settings(streaming_modules):
    engine = prepared_model(Options(temperature=0))
    captured = {}

    def generate(**kwargs):
        captured.update(kwargs)
        kwargs["streamer"].end()

    engine.model.generate = generate
    list(engine.stream([{"role": "user", "content": "Hi"}]))
    assert captured["do_sample"] is False
    assert captured["temperature"] is None


def test_load_uses_safe_files_and_pinned_revision(monkeypatch):
    captured = {}

    def from_pretrained(repository, **kwargs):
        captured.update(repository=repository, **kwargs)
        return SimpleNamespace(eval=lambda: None)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=from_pretrained),
            BitsAndBytesConfig=object,
        ),
    )
    engine = LocalModel(get_model(), Options(device="cpu", offline=True))
    engine.load()
    assert captured["trust_remote_code"] is False
    assert captured["use_safetensors"] is True
    assert captured["revision"] == get_model().revision
    assert captured["local_files_only"] is True
