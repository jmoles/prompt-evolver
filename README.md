# prompt-evolver

Evolutionary optimization of structured AI agent prompts using multi-objective genetic algorithms and local LLMs.

## Overview

prompt-evolver takes a structured markdown agent prompt, decomposes it into evolvable segments, and uses [pymoo](https://pymoo.org/) (NSGA-II) to evolve improved versions against a corpus of test scenarios. Fitness is evaluated using a 6-component multi-objective function that measures specificity, structural compliance, domain calibration, disposition match, actionability, and guideline compliance.

Runs fully local via [Ollama](https://ollama.com/) with no external API dependencies. Optional Claude API validation available via `--claude-validate`.

## Features

- **Segment-based genome** — Parses markdown prompts into sections (disposition, expertise, response style, etc.) that evolve independently while preserving overall structure
- **Multi-objective fitness** — 6 components: 3 deterministic (regex/keyword), 2 LLM-judged, 1 guideline-based. Includes length and tone drift penalties to prevent bloat.
- **Three mutation operators** — DE difference mutation (learns from population diversity), guideline injection (applies known-good patterns), and pruning (fights length drift)
- **Scenario-driven evaluation** — Test prompts against curated scenarios extracted from PDFs or written manually
- **Guidelines memory** — Persistent JSON database of discovered improvement patterns that accumulate across evolution runs
- **Local-first validation** — Triple-check pairwise comparison with majority vote. No API required.
- **Human-in-the-loop** — Evolution surfaces candidates; you review diffs and decide what to promote

## Installation

```bash
uv venv && uv pip install -e ".[dev]"
```

Requires [Ollama](https://ollama.com/) with a model installed:

```bash
ollama pull mistral:7b-instruct-v0.3-q5_K_M
```

## Usage

### Run evolution

```bash
python -m prompt_evolver.evolve evolve \
    --segment-schema config/segment_schema.yaml \
    --fitness-config config/fitness.yaml \
    --evolution-config config/evolution.yaml
```

### Compare two prompts

```bash
python -m prompt_evolver.evolve validate \
    --segment-schema config/segment_schema.yaml \
    --fitness-config config/fitness.yaml \
    --evolution-config config/evolution.yaml \
    --original original_agent.md \
    --evolved evolved_agent.md
```

### Extract scenarios from PDFs

```bash
python -m prompt_evolver.evolve scenarios \
    --segment-schema config/segment_schema.yaml \
    --fitness-config config/fitness.yaml \
    --evolution-config config/evolution.yaml \
    --pdfs doc1.pdf doc2.pdf \
    --output scenarios.json
```

### Manage guidelines

```bash
python -m prompt_evolver.evolve guidelines \
    --guidelines-path guidelines.json show
```

## Configuration

Three YAML files control everything. No domain-specific content is hardcoded in source.

| File | Purpose |
|------|---------|
| `config/segment_schema.yaml` | Defines expected prompt sections and header variants |
| `config/fitness.yaml` | Specificity patterns, calibration keywords, LLM rubrics, fitness weights |
| `config/evolution.yaml` | Population size, generations, operator weights, model, paths |

See `config/examples/` for a complete working example.

## How it works

1. **Parse** — Your agent prompt is split into segments matching `##` headers
2. **Initialize** — A population of variants is created from the seed prompt via light mutations
3. **Evaluate** — Each variant is scored against test scenarios using the 6-component fitness function
4. **Evolve** — NSGA-II selects, crosses over, and mutates the population over N generations
5. **Validate** — Top candidates are compared against the original via pairwise LLM judgment
6. **Review** — A diff report shows exactly what changed, segment by segment. You decide what to promote.

## Hardware

- **Minimum:** Any machine that can run a 7B model via Ollama
- **Recommended:** NVIDIA GPU with 8GB+ VRAM for ~3 min/generation
- **CPU-only:** Works but slower (~18 min/generation)
- **Typical run:** Population 16, 28 generations = ~1.5 hours on GPU

## Tests

```bash
python -m pytest tests/ -v
```

## License

[MIT](LICENSE)
