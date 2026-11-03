"""Local conversations with Rennervate."""

from collections.abc import Iterator
from dataclasses import dataclass
from importlib.util import find_spec
from math import isfinite
from queue import Empty
from threading import Event, Thread

from .models import Model


class RennervateError(Exception):
    """A problem that can be presented directly to the user."""


@dataclass(frozen=True)
class Options:
    device: str = "auto"
    compact: bool = False
    offline: bool = False
    cache_dir: str | None = None
    max_tokens: int = 512
    temperature: float = 0.7
    context_limit: int = 8192

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("Choose auto, cpu, cuda, or mps for the device.")
        if self.max_tokens < 1 or self.max_tokens >= self.context_limit:
            raise ValueError("The answer length must be positive and below the context limit.")
        if not isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2.")
        if self.compact and self.device not in {"auto", "cuda"}:
            raise ValueError("Compact mode requires a supported NVIDIA graphics card.")


def download(model: Model, options: Options) -> str:
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            model.repository,
            revision=model.revision,
            cache_dir=options.cache_dir,
            local_files_only=options.offline,
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "*.jinja",
                "LICENSE",
                "*NOTICE*",
                "README.md",
            ],
        )
    except Exception as exc:
        raise RennervateError(
            "The model download could not be completed. "
            "Check your connection and available storage."
        ) from exc


def _friendly_error(exc: Exception, options: Options) -> RennervateError:
    message = str(exc).lower()
    if isinstance(exc, MemoryError) or "out of memory" in message or "not enough memory" in message:
        return RennervateError(
            "There is not enough memory. Try the 4b model, --compact, or --device cpu."
        )
    if options.offline:
        return RennervateError(
            "The model could not be opened offline. Download it first with 'rennervate download' "
            "using the same model and cache directory."
        )
    return RennervateError(
        "The model could not be run. Check your connection, available storage, and memory. "
        "For help, contact team@banzhuan.ai."
    )


class LocalModel:
    def __init__(self, model: Model, options: Options):
        self.model_info = model
        self.options = options
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RennervateError(
                "Required packages are unavailable. Run 'pip install --upgrade rennervate'."
            ) from exc

        device = self.options.device
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise RennervateError("NVIDIA acceleration is unavailable. Try --device cpu.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RennervateError("Apple acceleration is unavailable. Try --device cpu.")
        if self.options.compact and (device != "cuda" or find_spec("bitsandbytes") is None):
            raise RennervateError(
                "Compact mode requires a supported NVIDIA graphics card and "
                "'pip install \"rennervate[compact]\"'."
            )

        common = {
            "revision": self.model_info.revision,
            "cache_dir": self.options.cache_dir,
            "local_files_only": self.options.offline,
            "trust_remote_code": False,
        }
        load_options = {"dtype": "auto", "use_safetensors": True}
        if device == "cuda" and self.options.device == "auto" and not self.options.compact:
            import psutil

            load_options["device_map"] = "auto"
            load_options["max_memory"] = {
                index: max(1, int(torch.cuda.mem_get_info(index)[0] * 0.75))
                for index in range(torch.cuda.device_count())
            }
            load_options["max_memory"]["cpu"] = int(psutil.virtual_memory().available * 0.8)
        else:
            load_options["device_map"] = {"": device}
        if device == "mps":
            load_options["dtype"] = torch.float16
        if self.options.compact:
            load_options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=(
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                ),
            )
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_info.repository, **common)
            loaded = AutoModelForCausalLM.from_pretrained(
                self.model_info.repository,
                **common,
                **load_options,
            )
            loaded.eval()
            self.tokenizer = tokenizer
            self.model = loaded
        except Exception as exc:
            raise _friendly_error(exc, self.options) from exc

    def _inputs(self, messages: list[dict[str, str]]):
        history = [dict(message) for message in messages]
        limit = min(self.options.context_limit, self.model.config.max_position_embeddings)
        while True:
            prompt = self.tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            if inputs["input_ids"].shape[-1] + self.options.max_tokens <= limit:
                return inputs.to(self.model.device)
            first = 1 if history[0]["role"] == "system" else 0
            if len(history) - first <= 1:
                raise RennervateError("This prompt is too long. Shorten it or reduce --max-tokens.")
            del history[first : first + 2]

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("A conversation must end with a user message.")
        self.load()
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        stop = Event()
        errors: list[Exception] = []

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return stop.is_set()

        inputs = self._inputs(messages)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=0.2,
        )
        settings = {
            "max_new_tokens": self.options.max_tokens,
            "do_sample": self.options.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([Cancelled()]),
        }
        if self.options.temperature > 0:
            settings.update(temperature=self.options.temperature, top_p=0.8, top_k=20)
        else:
            settings.update(temperature=None, top_p=None, top_k=None)

        def generate() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(**inputs, **settings)
            except Exception as exc:
                errors.append(exc)
                streamer.end()

        worker = Thread(target=generate, name="rennervate-generation", daemon=True)
        worker.start()
        try:
            while True:
                try:
                    chunk = next(streamer)
                except Empty:
                    if not worker.is_alive():
                        break
                    continue
                except StopIteration:
                    break
                if chunk:
                    yield chunk
        finally:
            stop.set()
            worker.join()
        if errors:
            raise _friendly_error(errors[0], self.options) from errors[0]
