# PrivateEdge

PrivateEdge is an on-device, real-time safety layer for livestreamers and content creators. It automatically detects and blocks sensitive, private, or harmful content before it ever reaches the stream.

## Problem

Describe the user-facing problem this project solves (privacy leaks, harmful content, compliance, etc.).

## Architecture

High-level layout:

| Path | Role |
|------|------|
| `src/capture/` | Video, screen, and audio capture |
| `src/models/` | Wrappers for ONNX + QNN Execution Provider |
| `src/policy/` | Rules, thresholds, emotion logic |
| `src/render/` | Masking, blurring, audio muting |
| `src/ui/` | Control panel |
| `src/configs/` | Thresholds and modes (source-side) |
| `models/` | ONNX weights and model configs |
| `configs/` | Thresholds and modes (deployment / root) |
| `scripts/` | Benchmarking and profiling |
| `tests/` | Unit tests for policy and basic I/O |

## Setup

Document prerequisites, environment, and how to obtain or place model files under `models/`.

## Usage

Document how to run the app, CLI flags, and configuration paths.

## Limitations

Document known constraints (platform support, latency, model size, false positives/negatives, etc.).
