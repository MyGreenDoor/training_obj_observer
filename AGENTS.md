# AGENTS.md

## Language & style
- Chat / explanations / summaries: **日本語**で回答すること．句読点は **「，．」**を優先すること．
- Code: **docstring とコメントは英語**で書くこと（既存のスタイルに合わせる）．
- Output format: 重要事項は箇条書きで簡潔に．必要に応じてファイルパスや関数名を明示すること．

## Repository overview (important paths)
- Entry script: `train_stereo_la.py` が学習のメイン実行スクリプト．
- Configs: `configs/` に設定ファイルを置く．
- Losses: `losses/` に損失関数関連モジュールを置く．
- Models: `models/` にネットワーク関連モジュールを置く．
- Utils: `utils/` に汎用モジュールを置く．
- Outputs: `outputs/` に学習ログや学習済みモデル（チェックポイント等）を置く．

## Critical constraint: external dataset repo
- Dataset classes are provided by a **separate repository**: `la_loader`.
- This repo may **not be available in cloud tasks** (e.g., Codex Cloud) and training will fail there.
- Therefore:
  - Prefer **local runs** for anything that requires dataset loading / training.
  - In cloud environments, limit work to **static analysis**, refactors, unit tests that do not require `la_loader`, and documentation.
  - If a task requires dataset execution, clearly state that it must be run locally and provide the exact local command(s).

## Expectations for architecture explanations
When asked to explain the codebase / architecture:
1. Start from `train_stereo_la.py` and describe:
   - main flow (config loading → model build → dataloader → loss → optimizer/scheduler → train loop → logging/checkpoint)
2. Summarize each top-level folder (`configs/`, `models/`, `losses/`, `utils/`, `outputs/`) with responsibilities.
3. List the key files referenced (paths) as evidence.
4. Call out assumptions and missing dependencies (`la_loader`) explicitly.

## Safe development workflow
- Always keep changes minimal and well-scoped.
- Prefer small, reviewable diffs.
- If making non-trivial changes:
  - Add/update lightweight checks (assertions, shape checks, simple unit tests) that do not require `la_loader` when possible.
- Never modify large binary artifacts under `outputs/` unless explicitly requested.
- Avoid writing secrets or tokens to files.

## Commands (guidance)
- For training / dataset-dependent runs: **local only** (requires `la_loader`).
- For cloud tasks: do not attempt full training. Prefer linting, type checks, and import-level tests that don't touch `la_loader`.

## If ambiguous
- Make the best effort with current info.
- State assumptions clearly and proceed with a safe, conservative plan.
