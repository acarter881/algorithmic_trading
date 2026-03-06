# CLAUDE.md

## Repository Scope

This repository is specifically for a Kalshi trading system targeting only these two series:

- `KXTOPMODEL`
- `KXLLM1`

Do not expand scope to other Kalshi markets, series, instruments, or generalized multi-market support unless the user explicitly asks for that change.

Assume all strategy, signal, validation, and execution work in this repo is for those two series only.

This codebase is financially sensitive. Default to safety-first behavior at all times.

## Market Context

As of March 2026, the two target series have approximately 39 total contracts:

- **KXTOPMODEL**: ~23 contracts (one per AI model + an "Other" catch-all)
- **KXLLM1**: ~16 contracts (one per AI company/org + an "Other" catch-all)

The exact count varies as Kalshi adds or removes events. Markets are dynamically discovered at startup via `TradingLoop._bootstrap_market_data()` / `_discover_markets()` using paginated API calls for each series.

### Signal Source

The primary signal source is the LMSYS Chatbot Arena leaderboard (arena.ai, fallback: lmarena.ai). The settlement metric for KXTOPMODEL contracts is Rank(UB) from the "Text, Overall, No Style Control" category. For KXLLM1, the winning organization is derived from the top model's org.

### Arena → Kalshi Contract Mapping

Arena model/org names must resolve to Kalshi contract tickers. The mapping chain is:

1. **Subtitle extraction** — contract names come from the Kalshi API `subtitle` field (or `yes_sub_title`, or parsed from `title`)
2. **Alias lookup** — `_DEFAULT_ORG_ALIASES` maps Arena org names to product names (e.g., "Anthropic" → "Claude") for KXLLM1
3. **Org name normalization** — `normalize_org_name()` strips common suffixes ("AI", "Labs", "DeepMind", "Research")
4. **Fuzzy matching** — `thefuzz.fuzz.token_sort_ratio` at 0.90 threshold
5. **Manual overrides** — `model_ticker_overrides` / `org_ticker_overrides` in `config/strategies/leaderboard_alpha.yaml`

The `_resolve_ticker()` method tries exact match first, then the original name, then stripped variants, before falling back to fuzzy matching. This handles common Arena→Kalshi discrepancies like "Google DeepMind" → "Google" or "DeepSeek Research" → "DeepSeek".

Mapping correctness is validated by `tests/unit/test_contract_mapping.py` against a fixture of all 39 contracts. If the test fails, the fix is usually to adjust `normalize_org_name`, add an alias, or add a manual override.

## Operating Mode

The system is designed for 24/7 paper trading operation via Docker (`docker-compose.yml` with `restart: unless-stopped`). Changes that could affect long-running stability — memory leaks, resource exhaustion, unbounded growth, error handling gaps — should be flagged.

## Default Operating Mode

Assume the following unless the user explicitly instructs otherwise:

- paper trading only
- local or development environment
- verification before claiming success
- minimal, reversible changes
- existing architecture should be preserved

Never enable live trading, real-money execution, or production credentials on your own.

## Core Rules

1. Make the smallest change that fully addresses the task.
2. Do not perform speculative refactors.
3. Do not silently change trading behavior, market-selection behavior, execution behavior, or risk behavior.
4. Do not weaken tests, type checks, or validation just to get a green result.
5. Do not ask the user to manually inspect something that can be checked through logs, tests, traces, API responses, or other machine-readable output.
6. Do not claim a task is complete unless the relevant validation passes.
7. If you are blocked, stop and report clearly instead of continuing to guess.

## Priority Order

When making decisions, optimize in this order:

1. correctness
2. safety
3. validation evidence
4. minimal scope
5. speed
6. style cleanup

## Task Execution Protocol

For every task, follow this process.

### 1. Understand the task

Before editing, identify:

- the smallest set of files likely involved
- the likely validation commands
- whether the task touches any high-risk trading logic

High-risk trading logic includes:

- signal generation
- strategy ranking or selection
- market selection
- order proposals
- order fills or rejections
- settlement logic
- position sizing
- risk controls
- timing logic tied to market state
- configuration that affects trading behavior

### 2. Inspect before editing

Read the relevant files before making changes.

Prefer extending existing patterns over introducing new abstractions.

Do not edit unrelated files unless they are directly required for the task.

### 3. Make minimal edits

Keep diffs tight and focused.

Avoid unrelated cleanup, broad renames, or architectural movement unless the task specifically requires it.

### 4. Validate immediately

After making a change, run the narrowest useful validation first, then broaden as needed.

Default validation for ordinary code changes:

- `ruff format .`
- `ruff check --fix .`
- `mypy src/autotrader/`
- `pytest -q`

For strategy, execution, market logic, or other high-risk changes, also run:

- `./scripts/docker_dev_loop.sh --iterations 1`

For broader or more consequential trading-logic changes, prefer:

- `./scripts/docker_dev_loop.sh --iterations 3`

If `scripts/agent_verify.sh` exists, use that as the default verification entry point for nontrivial changes.

### 5. Respond to validation results correctly

If validation fails:

- treat the concrete failure as the highest-priority truth
- fix the direct cause first
- rerun the exact failing command before doing anything else
- do not continue feature work while validation is red

If the same failure persists after two substantive fix attempts, stop and report.

## CI Policy

CI failures are blocking.

When CI reports a concrete error:

1. reproduce it locally if possible
2. fix the direct cause
3. rerun the exact failing command
4. confirm the failure is gone
5. only then resume broader work

Do not ignore CI failures.
Do not work around CI by disabling checks.
Do not describe work as complete while CI-relevant validation is still failing.

## Editing Rules

- Preserve the current architecture unless the task explicitly requires structural change.
- Reuse existing helpers and utilities where possible.
- Keep type annotations intact.
- Add or update tests for bug fixes when practical.
- Do not leave dead code, commented-out code, or placeholder TODOs unless explicitly requested.
- Do not rename or move modules unless necessary.
- Do not churn unrelated files.

## Trading Safety Rules

This repository is for `KXTOPMODEL` and `KXLLM1` only.

Treat any code touching the following as high risk:

- signal generation
- strategy output
- market eligibility
- proposal generation
- execution flow
- fill handling
- rejection handling
- settlement behavior
- sizing
- limits
- guardrails
- timing around market status

For high-risk changes:

- prefer very small diffs
- run broader validation
- summarize the expected behavior change explicitly
- do not infer correctness from code inspection alone

Never:

- enable live trading
- remove or weaken safeguards
- change risk defaults silently
- expand support to other Kalshi series without approval

## Debugging and Observability

Prefer machine-verifiable evidence over human interpretation.

When debugging, inspect:

- test output
- logs
- structured API responses
- traces
- local state
- database state
- generated artifacts

Do not ask the user for screenshots if the issue can be diagnosed from programmatic evidence.

## Stop Conditions

Stop and report instead of continuing when any of the following is true:

- the same validation failure persists after two substantive attempts
- a required environment variable, service, dependency, or secret is unavailable
- the repository state appears inconsistent or externally broken
- the task requires guessing user intent
- safe completion requires architectural clarification
- the requested change appears to widen scope beyond `KXTOPMODEL` or `KXLLM1`

## Response Format

When reporting completion, use this structure:

### Completed

- Files changed:
- What changed:
- Validation run:
- Remaining risk:

When reporting that you are blocked, use this structure:

### Blocked

- Files changed:
- What I ran:
- Exact failure:
- Likely root cause:
- Recommended next step:

## Things You Must Never Do

- never enable live trading on your own
- never expand scope beyond `KXTOPMODEL` and `KXLLM1` without explicit instruction
- never bypass validation to make progress appear faster
- never claim success without evidence
- never ask for manual review when machine-readable validation is available
- never keep thrashing on the same failure indefinitely
