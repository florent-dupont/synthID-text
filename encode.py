"""Generate plain and watermarked text with a local GGUF text model."""


import argparse
import json
import os
import random
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from decode import decode_payload, payload_slot, watermark_bit


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "models/model.gguf"


class ScoreResult(NamedTuple):
    value: float
    eligible_tokens: int


def validate_payload(payload: str) -> None:
    if not 1 <= len(payload) <= 8 or any(bit not in "01" for bit in payload):
        raise ValueError("payload must contain 1 to 8 bits, for example 1010")


def _validate_probabilities(probabilities: Sequence[float]) -> None:
    if not probabilities or any(p < 0 for p in probabilities):
        raise ValueError("probabilities must be nonempty and nonnegative")
    if sum(probabilities) <= 0:
        raise ValueError("at least one probability must be positive")


def sample_token(
    probabilities: Sequence[float],
    context: Sequence[int],
    key: bytes,
    layers: int,
    rng: random.Random,
    watermarked: bool = True,
    token_ids: Sequence[int] | None = None,
    payload: str | None = None,
) -> int:
    """Sample a token, optionally selecting it through a keyed tournament."""
    _validate_probabilities(probabilities)
    if not key:
        raise ValueError("key must not be empty")
    if not 1 <= layers <= 8:
        raise ValueError("layers must be between 1 and 8")
    if payload is not None:
        validate_payload(payload)
        if layers < 2:
            raise ValueError("a payload needs at least 2 tournament layers")

    vocabulary = list(token_ids) if token_ids is not None else list(range(len(probabilities)))
    if len(vocabulary) != len(probabilities) or any(token < 0 for token in vocabulary):
        raise ValueError("provide one nonnegative token ID per probability")
    if not watermarked:
        return rng.choices(vocabulary, weights=probabilities, k=1)[0]

    candidates = rng.choices(vocabulary, weights=probabilities, k=1 << layers)
    payload_target = 1
    if payload is not None:
        slot = payload_slot(key, context, len(payload))
        payload_target = int(payload[slot])
    for layer in range(layers):
        # Round 0 signals presence; later rounds repeat the assigned payload bit.
        target = payload_target if payload is not None and layer > 0 else 1
        winners = []
        for left, right in zip(candidates[::2], candidates[1::2]):
            left_bit = watermark_bit(key, context, layer, left)
            right_bit = watermark_bit(key, context, layer, right)
            if left_bit == right_bit:
                winners.append(rng.choice((left, right)))
            else:
                winners.append(left if left_bit == target else right)
        candidates = winners
    return candidates[0]


def score_details(
    tokens: Sequence[int],
    prompt: Sequence[int],
    key: bytes,
    layers: int,
    context_width: int = 4,
    presence_only: bool = False,
) -> ScoreResult:
    """Score unique contexts; with a payload, score only presence round 0."""
    if not tokens or not 1 <= layers <= 8 or context_width < 1:
        raise ValueError("provide tokens, 1 to 8 layers, and positive context width")
    history = list(prompt)
    total = 0
    eligible_tokens = 0
    seen_contexts = set()
    scored_layers = 1 if presence_only else layers
    for token in tokens:
        context = tuple(history[-context_width:])
        if context not in seen_contexts:
            total += sum(watermark_bit(key, context, layer, token) for layer in range(scored_layers))
            eligible_tokens += 1
            seen_contexts.add(context)
        history.append(token)
    return ScoreResult(total / (eligible_tokens * scored_layers), eligible_tokens)


def validate_context_budget(prompt_ids, max_tokens, n_ctx):
    """The context holds both the formatted prompt and generated tokens."""
    if not prompt_ids:
        raise ValueError("the prompt must contain at least one token")
    required = len(prompt_ids) + max_tokens
    if required > n_ctx:
        raise ValueError(
            f"prompt uses {len(prompt_ids)} tokens + {max_tokens} output tokens = "
            f"{required} tokens, but --n-ctx is {n_ctx}. "
            f"Increase --n-ctx to at least {required}."
        )


def resolve_ollama_model(name: str, models_dir: Path) -> Path:
    """Read a local Ollama manifest and locate its GGUF without an API call."""
    model_name, _, tag = name.partition(":")
    if not model_name or "/" in model_name or model_name in (".", ".."):
        raise ValueError("use a local library model name such as model:latest")
    tag = tag or "latest"
    if "/" in tag or tag in (".", ".."):
        raise ValueError("invalid model tag")
    manifest = models_dir / "manifests/registry.ollama.ai/library" / model_name / tag
    if not manifest.is_file():
        raise ValueError(
            f"local manifest not found: {manifest}. Set --models-dir to your "
            "Ollama models folder, or use --model-path for an existing GGUF."
        )
    data = json.loads(manifest.read_text())
    layers = [item for item in data["layers"] if item["mediaType"] == "application/vnd.ollama.image.model"]
    if len(layers) != 1:
        raise ValueError("expected one model weight file in the Ollama manifest")
    digest = layers[0]["digest"]
    if (
        not digest.startswith("sha256:")
        or len(digest) != 71
        or any(char not in "0123456789abcdef" for char in digest[7:])
    ):
        raise ValueError("invalid model digest")
    path = models_dir / "blobs" / digest.replace(":", "-")
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise ValueError("this model's weights are not a GGUF file")
    return path


def candidate_distribution(logits, temperature: float, top_k: int, top_p: float):
    """Convert the model's raw scores to a small next-token distribution."""
    import numpy as np

    if temperature <= 0 or top_k < 1 or not 0 < top_p <= 1:
        raise ValueError("use positive temperature/top_k and top_p in (0, 1]")
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("invalid model logits")
    count = min(top_k, values.size)
    ids = np.argpartition(values, -count)[-count:]
    ids = ids[np.argsort(-values[ids], kind="stable")]
    selected = values[ids]
    if not np.isfinite(selected[0]):
        raise ValueError("the model has no available next token")
    weights = np.exp((selected - selected[0]) / temperature)
    weights /= weights.sum()
    # Keep the token that crosses the cumulative top-p boundary too.
    keep = min(len(ids), int(np.searchsorted(np.cumsum(weights), top_p)) + 1)
    ids, weights = ids[:keep], weights[:keep]
    weights /= weights.sum()
    return ids.tolist(), weights.tolist()


def tokenize_prompt(model, text: str, raw_prompt: bool = False):
    if raw_prompt:
        return model.tokenize(text.encode(), add_bos=False, special=True)
    template = model.metadata.get("tokenizer.chat_template")
    if not template:
        return model.tokenize(text.encode(), add_bos=True, special=True)

    from jinja2 import TemplateError
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    def token_text(token):
        return model.detokenize([token], special=True).decode() if token >= 0 else ""

    try:
        formatter = Jinja2ChatFormatter(
            template=template,
            eos_token=token_text(model.token_eos()),
            bos_token=token_text(model.token_bos()),
        )
        prompt = formatter(
            messages=[{"role": "user", "content": text}], enable_thinking=False,
        ).prompt
    except (TemplateError, ValueError) as error:
        raise ValueError(
            "could not render the model's chat template; use --raw-prompt "
            "with a prompt formatted for this model"
        ) from error
    return model.tokenize(prompt.encode(), add_bos=False, special=True)


def generate_completion(model, prompt_ids, args, watermarked: bool):
    """Generate tokens, then score them using the same prompt context."""
    import llama_cpp
    import numpy as np

    model.reset()
    model.eval(prompt_ids)
    vocab = llama_cpp.llama_model_get_vocab(model.model)
    rng = random.Random(args.seed)
    key = args.key.encode()
    history = list(prompt_ids)
    tokens = []
    seen_contexts = set()
    started = time.perf_counter()
    stopped_on_eog = False
    for _ in range(args.max_tokens):
        pointer = llama_cpp.llama_get_logits_ith(model.ctx, -1)
        if not pointer:
            raise RuntimeError("model did not return next-token logits")
        logits = np.ctypeslib.as_array(pointer, shape=(model.n_vocab(),))
        token_ids, probabilities = candidate_distribution(
            logits, args.temperature, args.top_k, args.top_p
        )
        context = tuple(history[-args.context_width:])
        token = sample_token(
            probabilities, context, key, args.layers, rng,
            watermarked and context not in seen_contexts, token_ids=token_ids,
            payload=args.payload,
        )
        if llama_cpp.llama_vocab_is_eog(vocab, token):
            stopped_on_eog = True
            break
        tokens.append(token)
        history.append(token)
        seen_contexts.add(context)
        model.eval([token])
    result = {
        "text": model.detokenize(tokens, prev_tokens=prompt_ids).decode("utf-8", errors="replace"),
        "token_ids": tokens,
        "stopped_on_eog": stopped_on_eog,
        "generation_seconds": round(time.perf_counter() - started, 3),
    }
    if tokens:
        details = score_details(
            tokens, prompt_ids, key, args.layers, args.context_width,
            presence_only=args.payload is not None,
        )
        result["score"] = details.value
        result["eligible_tokens"] = details.eligible_tokens
        result["wrong_key_score"] = score_details(
            tokens, prompt_ids, b"different-demo-key", args.layers, args.context_width,
            presence_only=args.payload is not None,
        ).value
    else:
        result.update(score=None, eligible_tokens=0, wrong_key_score=None)
    if args.payload is not None:
        decoded = decode_payload(
            tokens, prompt_ids, key, len(args.payload), args.layers, args.context_width
        )
        result["payload"] = decoded._asdict()
    return result


def read_prompt(args):
    if args.prompt_file:
        text = args.prompt_file.read_text(encoding="utf-8")
    elif args.prompt is not None:
        text = args.prompt
    elif sys.stdin.isatty():
        text = input("Enter your prompt: ")
    else:
        text = sys.stdin.read()
    if not text.strip():
        raise ValueError("enter a prompt or provide --prompt or --prompt-file")
    return text


def completion_lines(result):
    value = result["score"]
    score_text = f"{value:.3f}" if value is not None else "no generated tokens"
    lines = [
        result["text"],
        f"Raw watermark score: {score_text} ({result['eligible_tokens']} eligible tokens)",
    ]
    if "payload" in result:
        lines.append(f"Payload estimate: {result['payload']['bits']}")
    return lines


def save_results(path, payload):
    """Write token data as JSON and completions as plain text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    sections = ["PROMPT", payload["settings"]["prompt"]]
    if payload["settings"].get("payload") is not None:
        sections.extend([
            "REQUESTED PAYLOAD", payload["settings"]["payload"],
            "Scores use presence round 0.",
        ])
    for label, result in payload["results"].items():
        sections.append(f"{label.upper()} TEXT")
        sections.extend(completion_lines(result))
    if payload["settings"].get("payload") is not None:
        sections.append("? means missing or tied votes. Bit estimates can be wrong.")
    text_path = path.with_name(path.stem + "_texts.txt")
    text_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return text_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--ollama-model", help="read an existing Ollama model, e.g. model:latest")
    source.add_argument("--model-path", type=Path, help=f"GGUF path; defaults to {DEFAULT_MODEL_PATH}")
    parser.add_argument(
        "--models-dir", type=Path,
        default=Path(os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama/models"))),
        help="Ollama model storage folder; also reads OLLAMA_MODELS",
    )
    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument("--prompt", help="prompt text; asks interactively when omitted")
    prompt_source.add_argument("--prompt-file", type=Path, help="read a UTF-8 prompt file")
    parser.add_argument("--raw-prompt", action="store_true", help="use the prompt as-is, skipping the model's chat template")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--context-width", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--key", default="portfolio-demo-key")
    parser.add_argument("--payload", help="embed 1 to 8 bits, e.g. 1010; requires at least 2 layers")
    parser.add_argument("--n-ctx", type=int, default=4096, help="total prompt + output capacity (default: 4096 tokens)")
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--output", type=Path, help="JSON path; defaults to a new file per run")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.context_width < 1 or args.n_ctx < 1 or not 1 <= args.layers <= 8:
        parser.error("use positive max-tokens/context-width/n-ctx and 1 to 8 layers")
    if not 1 <= len(args.key.encode()) <= 64:
        parser.error("key must contain 1 to 64 bytes")
    if args.temperature <= 0 or args.top_k < 1 or not 0 < args.top_p <= 1:
        parser.error("invalid sampling settings")
    try:
        if args.payload is not None:
            validate_payload(args.payload)
            if args.layers < 2:
                raise ValueError("--payload requires at least 2 layers")
        args.prompt = read_prompt(args)
        if args.ollama_model:
            path = resolve_ollama_model(args.ollama_model, args.models_dir)
        else:
            path = args.model_path or DEFAULT_MODEL_PATH
        if not path.is_file():
            raise ValueError(f"model file not found: {path}. Select a GGUF with --model-path.")
    except (ValueError, OSError, EOFError) as error:
        parser.error(str(error))
    if args.output is None:
        args.output = (ROOT / "results") / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    return parser, args, path


def main():
    parser, args, path = parse_args()
    try:
        import llama_cpp
    except ImportError:
        parser.error("local generation requires llama-cpp-python; follow the runtime setup in README.md")
    print(f"Loading {args.ollama_model or path.name}...", flush=True)
    model = llama_cpp.Llama(
        model_path=str(path), n_ctx=args.n_ctx, n_batch=128,
        n_gpu_layers=args.gpu_layers, logits_all=False, verbose=False,
    )
    try:
        try:
            prompt_ids = tokenize_prompt(model, args.prompt, args.raw_prompt)
            validate_context_budget(prompt_ids, args.max_tokens, args.n_ctx)
        except ValueError as error:
            parser.error(str(error))
        if args.payload is not None:
            print(f"Payload: {args.payload}; presence scores use round 0.")
        results = {}
        for label, marked in (("plain", False), ("watermarked", True)):
            print(f"\n{label.upper()} TEXT\nGenerating...", flush=True)
            results[label] = generate_completion(model, prompt_ids, args, marked)
            print("\n".join(completion_lines(results[label])), flush=True)
        settings = {
            name: value for name, value in vars(args).items()
            if name not in ("models_dir", "model_path", "output", "prompt_file")
        }
        payload = {
            "model_path": str(path), "settings": settings,
            "runtime": {"llama_cpp_python": llama_cpp.__version__},
            "prompt_token_ids": prompt_ids, "results": results,
            "score_mode": "presence_round_0" if args.payload is not None else "all_rounds",
            "payload_scheme": "context-buckets-v1" if args.payload is not None else None,
        }
        text_path = save_results(args.output, payload)
        print(f"\nTexts: {text_path}")
        print(f"Token data: {args.output}")
    finally:
        model.close()


if __name__ == "__main__":
    main()
