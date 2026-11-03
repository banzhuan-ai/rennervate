"""Available Rennervate models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    key: str
    name: str
    repository: str
    revision: str


MODELS = (
    Model(
        "4b",
        "Rennervate 4B Instruct",
        "BanZhuan-AI/Rennervate-Qwen3-4B-Instruct-2507-Refusal-Suppressed",
        "f7337bf51c9a6ba34c946151501fb1eba71e0aa6",
    ),
    Model(
        "8b",
        "Rennervate 8B",
        "BanZhuan-AI/Rennervate-Qwen3-8B-Refusal-Suppressed",
        "652ecec605990177df1e92cef1942a4a8a347904",
    ),
)


def get_model(name: str = "4b") -> Model:
    for model in MODELS:
        if name.casefold() in (model.key, model.repository.casefold()):
            return model
    raise ValueError("Choose a supported model: 4b or 8b.")
