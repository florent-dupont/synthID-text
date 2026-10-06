"""Decode a binary payload from saved token IDs."""


import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple


class PayloadResult(NamedTuple):
    bits: str
    zero_votes: list[int]
    one_votes: list[int]


def payload_slot(key: bytes, context: Sequence[int], payload_length: int) -> int:
    """Assign a context to a bit position, independently of the chosen token."""
    if not 1 <= payload_length <= 8:
        raise ValueError("payload length must be between 1 and 8")
    # The prefix separates this hash from watermark_bit's hash inputs.
    data = b"payload-slot|" + ",".join(map(str, context)).encode("ascii")
    digest = hashlib.blake2b(data, key=key, digest_size=8).digest()
    return int.from_bytes(digest, "big") % payload_length


def watermark_bit(
    key: bytes, context: Sequence[int], layer: int, token: int
) -> int:
    """Return a reproducible keyed bit for a candidate token and its context."""
    if not key:
        raise ValueError("key must not be empty")
    if layer < 0 or token < 0 or any(item < 0 for item in context):
        raise ValueError("layers and token IDs must be nonnegative")
    payload = ",".join(map(str, context)).encode("ascii")
    payload += b"|" + str(layer).encode("ascii")
    payload += b"|" + str(token).encode("ascii")
    digest = hashlib.blake2b(payload, key=key, digest_size=1).digest()
    return digest[0] & 1


def decode_payload(
    tokens: Sequence[int], prompt: Sequence[int], key: bytes,
    payload_length: int, layers: int, context_width: int = 4,
) -> PayloadResult:
    """Decode by majority vote over payload rounds; ties return '?'."""
    if not key or not 1 <= len(key) <= 64:
        raise ValueError("key must contain 1 to 64 bytes")
    if not 1 <= payload_length <= 8 or not 2 <= layers <= 8 or context_width < 1:
        raise ValueError("use 1 to 8 payload bits, 2 to 8 layers, and positive context width")
    zero_votes = [0] * payload_length
    one_votes = [0] * payload_length
    history = list(prompt)
    seen_contexts = set()
    for token in tokens:
        context = tuple(history[-context_width:])
        if context not in seen_contexts:
            slot = payload_slot(key, context, payload_length)
            for layer in range(1, layers):
                bit = watermark_bit(key, context, layer, token)
                if bit == 1:
                    one_votes[slot] += 1
                else:
                    zero_votes[slot] += 1
            seen_contexts.add(context)
        history.append(token)
    bits = []
    for zeros, ones in zip(zero_votes, one_votes):
        if ones > zeros:
            bits.append("1")
        elif zeros > ones:
            bits.append("0")
        else:
            bits.append("?")
    return PayloadResult("".join(bits), zero_votes, one_votes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON saved by encode.py")
    parser.add_argument("--payload-length", type=int, required=True, help="number of embedded bits (1–8)")
    parser.add_argument("--key", default="portfolio-demo-key", help="key used during generation")
    args = parser.parse_args()
    try:
        run = json.loads(args.input.read_text(encoding="utf-8"))
        if run.get("payload_scheme") != "context-buckets-v1":
            raise ValueError("select a run generated with --payload using context-buckets-v1")
        settings = run["settings"]
        for label in ("plain", "watermarked"):
            decoded = decode_payload(
                run["results"][label]["token_ids"], run["prompt_token_ids"],
                args.key.encode(), args.payload_length, settings["layers"],
                settings["context_width"],
            )
            print(f"{label.capitalize()} payload estimate: {decoded.bits}")
            print(f"  Zero votes: {decoded.zero_votes}")
            print(f"  One votes:  {decoded.one_votes}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print("? means missing or tied votes. An estimate does not prove watermark presence.")


if __name__ == "__main__":
    main()
