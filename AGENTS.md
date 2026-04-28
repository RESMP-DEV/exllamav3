# ExLlamaV3 — Repository Guide

## Project Scope

ExLlamaV3 is a local LLM inference library for modern consumer GPUs. It focuses on EXL3 quantization, inference kernels, dynamic batching, multimodal support, and interoperability with ecosystem tools such as TabbyAPI and Transformers.

## Environment and Commands

This repo supports wheel install, PyPI install, or source build. For repository work, prefer the source path.

Common workflows from the README:

- install PyTorch separately for a supported CUDA version,
- install repo requirements,
- run conversion with `python convert.py ...`,
- run example scripts such as `python examples/chat.py ...`.

When changing source code, validate with the relevant examples or tests under `tests/`.

## Repository Layout

Important areas:

- `exllamav3/` — main library code.
- `examples/` — inference and integration examples.
- `eval/` — evaluation helpers.
- `doc/` — EXL3 and conversion documentation.
- `tests/` — test coverage.
- `util/` and `science/` — supporting scripts and experimental work.

## Working Rules

Keep these invariants intact:

- EXL3 conversion and inference paths should remain aligned with the checked-in docs,
- TabbyAPI remains the recommended OpenAI-compatible server pairing,
- optional high-performance paths such as Flash Linear Attention should remain optional rather than hard requirements,
- and model-architecture support should be extended consistently across docs, code, and examples.

If you change installation or conversion behavior, update the corresponding documentation in `doc/` or the README.

## Validation Expectations

Prefer targeted validation:

- conversion changes should exercise `convert.py`,
- runtime or backend changes should exercise a focused example or test,
- and architecture-support changes should update the relevant support lists and example assumptions.

## Change log expectations

Update `CHANGELOG.md` when a change affects model support, conversion behavior, runtime behavior, installation guidance, or validation expectations. Keep entries concise and summarize the effect of upstream syncs instead of mirroring commit history.
