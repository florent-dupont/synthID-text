# SynthID-Text

A local experiment with SynthID-style tournament sampling. `encode.py` generates
plain and watermarked text, optionally carrying a short binary payload.
`decode.py` recovers the bits from the saved token IDs.

## Model file

Put a GGUF text-generation model supported by
[llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/models.md)
in a `models` folder next to the scripts. Rename it to `model.gguf`:

```text
models/model.gguf
```

To keep its original name or store it elsewhere, pass
`--model-path /path/to/your-model.gguf`. Compatibility depends on the model
architecture and your installed runtime, not just the file extension.

## Setup

Use Python 3.12 and a C++ build toolchain. On macOS with Apple Silicon:

```sh
python3.12 -m venv .venv
CMAKE_ARGS=-DGGML_METAL=on .venv/bin/python -m pip install "llama-cpp-python>=0.3.36" "numpy>=2,<3"
```

For a CPU build, use `CMAKE_ARGS=-DGGML_METAL=off`. New model architectures
may require a newer `llama-cpp-python` version.

## Encode

```sh
.venv/bin/python encode.py \
  --prompt "Describe a floating city with unusual gardens and transport." \
  --payload 1010 \
  --max-tokens 256 \
  --output results/payload.json
```

Omit `--prompt` to enter it interactively. Omit `--payload` for watermarking
without a message. Payloads accept one to eight binary digits. Both completions
are printed and saved as JSON plus a readable text file. `--n-ctx` must fit the
prompt and output together; its default is 4096. Use `--gpu-layers 0` for CPU
execution. `encode.py --help` lists all options.

The encoder uses the GGUF's embedded chat template when available, or plain
text when there is no template. `--raw-prompt` skips formatting so you can
supply a prompt in the format your model expects.

## Decode

```sh
python3.12 decode.py results/payload.json --payload-length 4
```

The decoder needs no model or extra dependencies. If encoding used a custom key,
pass the same `--key` here. It reads the saved prompt and output token IDs rather
than the stored requested bits. A `?` means missing or tied votes.

This is an independent experiment based on the
[SynthID-Text paper](https://www.nature.com/articles/s41586-024-08025-4).
Scores are uncalibrated. Payload recovery requires the original saved tokens;
it does not yet handle copied or edited text, and a bit estimate alone does not
prove watermark presence. Model files and generated results stay local.
