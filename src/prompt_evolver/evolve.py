"""Main CLI entry point for prompt-evolver."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.optimize import minimize

from prompt_evolver.analysis import EvolutionAnalyzer
from prompt_evolver.config import EvolverConfig, load_config
from prompt_evolver.fitness import FitnessEvaluator, FitnessResult
from prompt_evolver.genome import Genome, parse_markdown, render_markdown
from prompt_evolver.guidelines import GuidelinesDB
from prompt_evolver.llm import LLMClient
from prompt_evolver.operators import (
    PromptDuplicateElimination,
    PromptMutation,
    PromptSampling,
    SegmentCrossover,
)
from prompt_evolver.problem import PromptEvolutionProblem
from prompt_evolver.scenarios import ScenarioExtractor
from prompt_evolver.validator import Validator


class EvolutionCallback(Callback):
    """Per-generation logging and checkpointing with progress tracking."""

    def __init__(self, analyzer: EvolutionAnalyzer, evaluator: FitnessEvaluator, config: EvolverConfig):
        super().__init__()
        self.analyzer = analyzer
        self.evaluator = evaluator
        self.config = config
        self.total_gens = config.evolution.n_generations
        self.start_time = time.time()
        self.gen_times: list[float] = []

    def notify(self, algorithm):
        gen = algorithm.n_gen
        now = time.time()
        pop = algorithm.pop

        # Track generation timing
        if self.gen_times:
            gen_duration = now - self.gen_times[-1]
        else:
            gen_duration = now - self.start_time
        self.gen_times.append(now)

        # Estimate remaining time
        elapsed = now - self.start_time
        avg_per_gen = elapsed / gen if gen > 0 else 0
        remaining_gens = self.total_gens - gen
        eta_seconds = avg_per_gen * remaining_gens
        eta_str = _format_duration(eta_seconds)
        elapsed_str = _format_duration(elapsed)
        pct = (gen / self.total_gens) * 100

        # Extract fitness results from population
        fitness_results = []
        for ind in pop:
            F = ind.F
            w = self.config.fitness.weights
            weights = [w.specificity, w.structure, w.calibration,
                       w.disposition, w.actionability, w.guideline_compliance]
            scores = [-F[i] / max(weights[i], 1e-9) for i in range(6)]
            fitness_results.append(FitnessResult(*[max(0, min(1, s)) for s in scores]))

        self.analyzer.track_generation(gen, fitness_results)

        # Compute population best and mean for each objective
        best = FitnessResult(*[
            max(r.as_tuple()[i] for r in fitness_results) for i in range(6)
        ])

        # Checkpoint
        if self.config.evolution.save_every_gen:
            best_genome = pop[0].X[0]
            self.analyzer.save_genome(best_genome, f"gen_{gen:03d}_best.md")

        # Progress bar
        bar_width = 20
        filled = int(bar_width * gen / self.total_gens)
        bar = "█" * filled + "░" * (bar_width - filled)

        print(f"  [{bar}] {gen}/{self.total_gens} ({pct:.0f}%) | "
              f"{_format_duration(gen_duration)}/gen | "
              f"elapsed {elapsed_str} | eta {eta_str}")
        print(f"    best: spec={best.specificity:.2f} "
              f"struct={best.structure:.2f} "
              f"cal={best.calibration:.2f} "
              f"disp={best.disposition:.2f} "
              f"act={best.actionability:.2f} "
              f"guide={best.guideline_compliance:.2f}")


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h{m:02d}m"


def run_evolution(config: EvolverConfig) -> None:
    """Main evolution loop."""
    print("=== prompt-evolver: starting evolution ===\n")

    # Load seed genome
    seed_path = config.evolution.seed_profile_path
    if not seed_path or not Path(seed_path).exists():
        print(f"Error: seed profile not found: {seed_path}", file=sys.stderr)
        sys.exit(1)

    with open(seed_path) as f:
        seed_text = f.read()
    seed_genome = parse_markdown(seed_text, config.segments)
    print(f"Loaded seed: {seed_path} ({len(seed_genome.segments)} segments)")

    # Load scenarios
    scenarios = []
    if config.evolution.scenario_path and Path(config.evolution.scenario_path).exists():
        with open(config.evolution.scenario_path) as f:
            scenarios = json.load(f)
    print(f"Loaded {len(scenarios)} scenarios")

    if not scenarios:
        print("Warning: no scenarios loaded. Fitness evaluation will be limited.",
              file=sys.stderr)

    # Load guidelines
    guidelines_db = GuidelinesDB(config.evolution.guidelines_path or "guidelines.json")
    guidelines_db.load()
    print(f"Loaded {len(guidelines_db.guidelines)} guidelines")

    # Initialize LLM client
    llm = LLMClient(
        model=config.evolution.llm_model,
        temperature=config.evolution.llm_temperature,
        max_tokens=config.evolution.llm_max_tokens,
        timeout=config.evolution.llm_timeout,
    )

    # Initialize fitness evaluator (with reference genome for length/tone penalties)
    segment_keys = [s.key for s in config.segments]
    evaluator = FitnessEvaluator(
        config=config.fitness,
        llm=llm,
        segment_keys=segment_keys,
        guidelines=guidelines_db.get_all(),
        reference_genome=seed_genome,
    )

    # Initialize operators
    sampling = PromptSampling(seed_genome, llm)
    mutation = PromptMutation(
        llm=llm,
        guidelines_db=guidelines_db,
        de_diff_weight=config.evolution.mutation_de_diff_weight,
        guideline_weight=config.evolution.mutation_guideline_weight,
        prune_weight=config.evolution.mutation_prune_weight,
    )
    crossover = SegmentCrossover(swap_prob=config.evolution.crossover_prob)

    # Initialize problem
    problem = PromptEvolutionProblem(
        fitness_evaluator=evaluator,
        scenarios=scenarios,
        config=config,
    )

    # Initialize analyzer
    analyzer = EvolutionAnalyzer(config.evolution.output_dir)

    # Configure NSGA-II
    algorithm = NSGA2(
        pop_size=config.evolution.pop_size,
        sampling=sampling,
        mutation=mutation,
        crossover=crossover,
        eliminate_duplicates=PromptDuplicateElimination(),
    )

    callback = EvolutionCallback(analyzer, evaluator, config)

    print(f"\nRunning NSGA-II: pop={config.evolution.pop_size}, "
          f"gens={config.evolution.n_generations}\n")

    # Run evolution
    result = minimize(
        problem,
        algorithm,
        ("n_gen", config.evolution.n_generations),
        callback=callback,
        verbose=False,
    )

    # Extract best from Pareto front
    print(f"\n=== Evolution complete. Pareto front: {len(result.opt)} solutions ===\n")

    if len(result.opt) > 0:
        best_genome = result.opt[0].X[0]
        analyzer.save_genome(best_genome, "best_evolved.md")
        analyzer.save_genome(seed_genome, "original.md")

        # Generate diff
        diff = analyzer.segment_diff_report(seed_genome, best_genome)
        diff_path = Path(config.evolution.output_dir) / "diff_report.md"
        with open(diff_path, "w") as f:
            f.write(diff)
        print(f"Diff report: {diff_path}")

        # Run validation
        validator = Validator(llm, n_rounds=config.evolution.validation_rounds)

        print("\nRunning local validation...")
        val_result = validator.validate(seed_genome, best_genome, scenarios)
        print(f"  Winner: {val_result.winner} ({val_result.confidence})")
        print(f"  Votes: {val_result.votes}")

        # Optional Claude validation
        if config.evolution.validation_mode in ("claude", "both"):
            print("\nRunning Claude validation...")
            claude_result = validator.validate(
                seed_genome, best_genome, scenarios,
                model_override=config.evolution.validation_model,
            )
            print(f"  Claude winner: {claude_result.winner} ({claude_result.confidence})")

        # Extract guidelines from the best candidate
        print("\nExtracting guidelines from best candidate...")
        new_guidelines = analyzer.extract_guidelines(seed_genome, best_genome, llm)
        for g in new_guidelines:
            guidelines_db.add(
                text=g.get("text", ""),
                segment_key=g.get("segment_key", ""),
                source="evolution",
                keywords=g.get("keywords", []),
                polarity=g.get("polarity", "positive"),
            )
        guidelines_db.save()
        print(f"  Added {len(new_guidelines)} new guidelines")

    # Save convergence log
    analyzer.save_convergence_log()
    convergence = analyzer.convergence_report()
    conv_path = Path(config.evolution.output_dir) / "convergence_report.md"
    with open(conv_path, "w") as f:
        f.write(convergence)
    print(f"\nConvergence report: {conv_path}")
    print("\n=== Done ===")


def run_validate(config: EvolverConfig, original_path: str, evolved_path: str) -> None:
    """Standalone validation of two profiles."""
    with open(original_path) as f:
        original = parse_markdown(f.read(), config.segments)
    with open(evolved_path) as f:
        evolved = parse_markdown(f.read(), config.segments)

    llm = LLMClient(
        model=config.evolution.llm_model,
        temperature=config.evolution.llm_temperature,
    )

    scenarios = []
    if config.evolution.scenario_path and Path(config.evolution.scenario_path).exists():
        with open(config.evolution.scenario_path) as f:
            scenarios = json.load(f)

    validator = Validator(llm, n_rounds=config.evolution.validation_rounds)

    print("Running validation...")
    result = validator.validate(original, evolved, scenarios)
    print(f"Winner: {result.winner} ({result.confidence})")
    print(f"Votes: {result.votes}")
    print(f"\n{result.diff_summary}")

    if config.evolution.validation_mode in ("claude", "both"):
        print("\nRunning Claude validation...")
        claude_result = validator.validate(
            original, evolved, scenarios,
            model_override=config.evolution.validation_model,
        )
        print(f"Claude winner: {claude_result.winner} ({claude_result.confidence})")


def run_scenarios(config: EvolverConfig, pdf_paths: list[str], output: str, n_scenarios: int) -> None:
    """Extract and curate scenarios from PDFs."""
    llm = LLMClient(
        model=config.evolution.llm_model,
        temperature=config.evolution.llm_temperature,
    )
    extractor = ScenarioExtractor(llm)

    all_texts = []
    for pdf_path in pdf_paths:
        print(f"Extracting text from {pdf_path}...")
        chunks = extractor.extract_from_pdf(pdf_path)
        all_texts.extend(chunks)
        print(f"  {len(chunks)} pages extracted")

    print(f"\nCurating {n_scenarios} scenarios...")
    segment_keys = [s.key for s in config.segments]
    scenarios = extractor.curate_scenarios(all_texts, segment_keys, n_scenarios)

    extractor.save_scenarios(scenarios, output)
    print(f"Saved {len(scenarios)} scenarios to {output}")


def run_guidelines(guidelines_path: str, action: str, config: EvolverConfig | None = None) -> None:
    """Manage the guidelines database."""
    db = GuidelinesDB(guidelines_path)
    db.load()

    if action == "show":
        print(db.export_summary())
    elif action == "export":
        print(json.dumps(db.get_all(), indent=2))
    elif action == "consolidate":
        if not config:
            print("Error: --evolution-config required for consolidation", file=sys.stderr)
            sys.exit(1)
        llm = LLMClient(model=config.evolution.llm_model)
        print(f"Before: {len(db.guidelines)} guidelines")
        # Simple consolidation: remove low-confidence, low-seen guidelines
        db.guidelines = [
            g for g in db.guidelines
            if g.confidence >= 0.3 or g.times_seen >= 2
        ]
        db.save()
        print(f"After: {len(db.guidelines)} guidelines")


def build_cli() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="prompt-evolver",
        description="Evolutionary optimization of structured agent prompts",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common args
    def add_config_args(sub):
        sub.add_argument("--segment-schema", required=True, help="Path to segment_schema.yaml")
        sub.add_argument("--fitness-config", required=True, help="Path to fitness.yaml")
        sub.add_argument("--evolution-config", required=True, help="Path to evolution.yaml")

    # evolve
    evolve_parser = subparsers.add_parser("evolve", help="Run evolutionary optimization")
    add_config_args(evolve_parser)
    evolve_parser.add_argument("--claude-validate", action="store_true",
                               help="Use Claude API for validation alongside local")
    evolve_parser.add_argument("--verbose", action="store_true")

    # validate
    validate_parser = subparsers.add_parser("validate", help="Compare two profiles")
    add_config_args(validate_parser)
    validate_parser.add_argument("--original", required=True, help="Path to original profile")
    validate_parser.add_argument("--evolved", required=True, help="Path to evolved profile")
    validate_parser.add_argument("--claude-validate", action="store_true")

    # scenarios
    scenarios_parser = subparsers.add_parser("scenarios", help="Extract scenarios from PDFs")
    add_config_args(scenarios_parser)
    scenarios_parser.add_argument("--pdfs", nargs="+", required=True, help="PDF file paths")
    scenarios_parser.add_argument("--output", required=True, help="Output JSON path")
    scenarios_parser.add_argument("--n-scenarios", type=int, default=20)

    # guidelines
    guidelines_parser = subparsers.add_parser("guidelines", help="Manage guidelines DB")
    guidelines_parser.add_argument("--guidelines-path", required=True)
    guidelines_parser.add_argument("action", choices=["show", "consolidate", "export"])
    guidelines_parser.add_argument("--segment-schema", help="Path to segment_schema.yaml")
    guidelines_parser.add_argument("--fitness-config", help="Path to fitness.yaml")
    guidelines_parser.add_argument("--evolution-config", help="Path to evolution.yaml")

    return parser


def main():
    """CLI entry point."""
    parser = build_cli()
    args = parser.parse_args()

    if args.command == "guidelines":
        config = None
        if args.evolution_config and args.segment_schema and args.fitness_config:
            config = load_config(args.segment_schema, args.fitness_config, args.evolution_config)
        run_guidelines(args.guidelines_path, args.action, config)
        return

    config = load_config(args.segment_schema, args.fitness_config, args.evolution_config)

    if hasattr(args, "claude_validate") and args.claude_validate:
        config.evolution.validation_mode = "both"

    if args.command == "evolve":
        run_evolution(config)
    elif args.command == "validate":
        run_validate(config, args.original, args.evolved)
    elif args.command == "scenarios":
        run_scenarios(config, args.pdfs, args.output, args.n_scenarios)


if __name__ == "__main__":
    main()
