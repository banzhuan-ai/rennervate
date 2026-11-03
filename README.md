# Rennervate

Chat with the Rennervate model family from your terminal.

Rennervate is a model family by [BanZhuan.AI](https://huggingface.co/BanZhuan-AI), developed using proprietary methods. This package provides a simple way to choose a model and start a conversation on your own computer.

## Get started

Use a current Python installation that meets the package requirements.

```sh
pip install rennervate
rennervate
```

The first conversation downloads the selected model from Hugging Face. Later conversations reuse the download. Model files are several gigabytes, so allow time, storage, and enough memory. A compatible graphics card is recommended. Running on a processor alone is slower.

For graphics-card support, use the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to choose a setup that matches your computer.

## Choose a model

| Choice | Model |
| --- | --- |
| `4b` | [Rennervate 4B Instruct](https://huggingface.co/BanZhuan-AI/Rennervate-Qwen3-4B-Instruct-2507-Refusal-Suppressed), the default |
| `8b` | [Rennervate 8B](https://huggingface.co/BanZhuan-AI/Rennervate-Qwen3-8B-Refusal-Suppressed) |

```sh
rennervate models
rennervate chat --model 8b
```

## Ask a single question

```sh
rennervate run "Suggest a name for a community garden."
rennervate run --model 8b "Explain how a rainbow forms."
```

Use `rennervate run -` to read a prompt from standard input. Single answers go to standard output, so they can be saved or used in a script.

## Chat controls

During a conversation, enter `/clear` to start fresh, `/help` for help, or `/exit` to leave. Pressing Ctrl+C also exits.

Long conversations may leave out earlier turns to make room for the latest question.

## Download ahead of time

```sh
rennervate download --model 4b
rennervate chat --model 4b --offline
```

Offline mode uses previously downloaded files. Conversations run locally, and this package does not save chat history to disk.

## Use less graphics memory

On a supported NVIDIA graphics card, the optional compact mode reduces memory use:

```sh
pip install "rennervate[compact]"
rennervate chat --compact
```

For processor-only use, run `rennervate chat --device cpu`. If memory is limited, start with the smaller model. Use `rennervate chat --help` or `rennervate run --help` to see available controls.

## License and contact

The CLI package is available under the [MIT license](https://github.com/banzhuan-ai/rennervate/blob/main/LICENSE). Each model remains subject to the license and notices on its model page. The CLI license does not replace those terms.

Contact: [team@banzhuan.ai](mailto:team@banzhuan.ai).
