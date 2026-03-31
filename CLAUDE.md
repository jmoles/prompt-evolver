# prompt-evolver

Evolutionary optimization of structured AI agent prompts using pymoo (NSGA-II) and local LLMs via Ollama.

## What this tool does

Takes a structured markdown agent prompt (with ## sections like disposition, expertise, response style, etc.), evolves it against test scenarios using multi-objective optimization, and produces improved prompt candidates for human review.

Fully local by default — runs on Ollama with no external API dependencies. Optional Claude API validation via `--claude-validate` flag.

## Quick start

```bash
# Install
uv venv && uv pip install -e ".[dev]"

# Ensure Ollama is running with a model
ollama pull mistral:7b-instruct-v0.3-q5_K_M

# Run tests
python -m pytest tests/ -v

# Extract scenarios from source documents
python -m prompt_evolver.evolve scenarios \
    --segment-schema config/segment_schema.yaml \
    --fitness-config config/fitness.yaml \
    --evolution-config config/evolution.yaml \
    --pdfs doc1.pdf doc2.pdf \
    --output scenarios.json

# Run evolution
python -m prompt_evolver.evolve evolve \
    --segment-schema config/segment_schema.yaml \
    --fitness-config config/fitness.yaml \
    --evolution-config config/evolution.yaml

# Compare two prompts
python -m prompt_evolver.evolve validate \
    --segment-schema config/segment_schema.yaml \
    --fitness-config config/fitness.yaml \
    --evolution-config config/evolution.yaml \
    --original original_agent.md \
    --evolved evolved_agent.md

# View guidelines database
python -m prompt_evolver.evolve guidelines \
    --guidelines-path guidelines.json show
```

## Configuration

Three YAML files control everything. No domain-specific content is hardcoded in source.

- `config/segment_schema.yaml` — defines the expected ## sections of agent prompts and their header variants
- `config/fitness.yaml` — specificity regex patterns, calibration keywords, vague phrase penalties, LLM rubrics, fitness weights, and deterministic floors
- `config/evolution.yaml` — population size, generations, mutation operator weights, LLM model, scenario path, output directory

See `config/examples/` for a complete working example using a generic customer support agent.

## Architecture

- **Genome:** Agent prompts parsed into segments (disposition, expertise, response_style, etc.) that evolve independently
- **Fitness:** 6-component multi-objective: specificity (regex), structure (deterministic), calibration (keywords), disposition (LLM-judged), actionability (LLM-judged), guideline compliance (keywords). Includes length penalty and tone drift penalty to prevent bloat.
- **Operators:** DE difference mutation (60%), guideline injection (25%), pruning (15%), segment-level crossover
- **Selection:** NSGA-II via pymoo with Pareto front extraction
- **Validation:** Local triple-check pairwise comparison with majority vote. Optional Claude API via `--claude-validate`.

## Key files

- `src/prompt_evolver/evolve.py` — CLI entry point
- `src/prompt_evolver/genome.py` — markdown prompt parser/renderer
- `src/prompt_evolver/fitness.py` — 6-component fitness evaluator
- `src/prompt_evolver/operators.py` — mutation, crossover, sampling operators
- `src/prompt_evolver/problem.py` — pymoo Problem subclass
- `src/prompt_evolver/guidelines.py` — persistent guidelines memory (JSON)
- `src/prompt_evolver/scenarios.py` — PDF extraction and scenario curation
- `src/prompt_evolver/validator.py` — pairwise validation
- `src/prompt_evolver/analysis.py` — convergence tracking, diff reports

## How to customize for your agents

1. Write your agent prompt as a markdown file with ## sections
2. Update `segment_schema.yaml` with the header variants your prompt uses
3. Create domain-specific `fitness.yaml` with regex patterns that match concrete language in your domain (e.g., specific terminology, named entities, reference numbers)
4. Create scenarios (manually as JSON, or extract from PDFs with the `scenarios` command)
5. Optionally seed `guidelines.json` with known-good rules for your domain
6. Point `evolution.yaml` at your agent, scenarios, and guidelines
7. Run `evolve` and review the diff report

## Hardware requirements

- **Minimum:** Any machine that can run a 7B model via Ollama (8GB RAM, CPU-only works but slow)
- **Recommended:** NVIDIA GPU with 8GB+ VRAM. ~3 min/generation with GPU, ~18 min/generation CPU-only.
- **Population 16, 28 generations:** ~1.5 hours on GPU, ~8 hours on CPU

## Dependencies

All permissively licensed (MIT, Apache-2.0, BSD):
- pymoo — multi-objective optimization
- litellm — LLM abstraction (Ollama, Claude, etc.)
- pyyaml — config parsing
- pypdf — PDF text extraction
- numpy — required by pymoo
- pytest — testing
