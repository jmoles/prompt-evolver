"""Configuration loading and validation for prompt-evolver."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SegmentDef:
    """Definition of a single prompt segment."""

    key: str
    description: str
    header_variants: list[str]
    required: bool = True
    max_tokens: int = 500


@dataclass
class FitnessWeights:
    """Weights for each fitness objective."""

    specificity: float = 1.0
    structure: float = 1.0
    calibration: float = 1.0
    disposition: float = 0.5
    actionability: float = 0.8
    guideline_compliance: float = 0.6


@dataclass
class FitnessConfig:
    """Configuration for the fitness evaluator."""

    weights: FitnessWeights
    specificity_patterns: dict[str, list[str]] = field(default_factory=dict)
    calibration_keywords: dict[str, list[str]] = field(default_factory=dict)
    guideline_keywords: dict[str, list[str]] = field(default_factory=dict)
    deterministic_floors: dict[str, float] = field(default_factory=dict)
    vague_phrases: list[str] = field(default_factory=list)
    llm_prompts: dict[str, str] = field(default_factory=dict)


@dataclass
class EvolutionConfig:
    """Configuration for the evolutionary algorithm."""

    pop_size: int = 16
    n_generations: int = 28
    seed_profile_path: str = ""
    scenario_path: str = ""
    guidelines_path: str = ""
    output_dir: str = "output"
    llm_model: str = "ollama/mistral:7b-instruct-v0.3-q5_K_M"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024
    llm_timeout: int = 120
    mutation_de_diff_weight: float = 0.60
    mutation_guideline_weight: float = 0.25
    mutation_prune_weight: float = 0.15
    crossover_prob: float = 0.3
    validation_mode: str = "local"  # "local" | "claude" | "both"
    validation_model: str = "anthropic/claude-sonnet-4-20250514"
    validation_rounds: int = 3
    scenario_rotation: str = "round_robin"  # "round_robin" | "random"
    save_every_gen: bool = True
    consolidate_every: int = 5


@dataclass
class EvolverConfig:
    """Top-level configuration combining all sub-configs."""

    segments: list[SegmentDef]
    fitness: FitnessConfig
    evolution: EvolutionConfig


def _parse_segment_schema(data: dict) -> list[SegmentDef]:
    """Parse segment definitions from YAML data."""
    segments = []
    for s in data.get("segments", []):
        segments.append(
            SegmentDef(
                key=s["key"],
                description=s.get("description", ""),
                header_variants=s.get("header_variants", []),
                required=s.get("required", True),
                max_tokens=s.get("max_tokens", 500),
            )
        )
    return segments


def _parse_fitness_config(data: dict) -> FitnessConfig:
    """Parse fitness configuration from YAML data."""
    weights_data = data.get("weights", {})
    weights = FitnessWeights(
        specificity=weights_data.get("specificity", 1.0),
        structure=weights_data.get("structure", 1.0),
        calibration=weights_data.get("calibration", 1.0),
        disposition=weights_data.get("disposition", 0.5),
        actionability=weights_data.get("actionability", 0.8),
        guideline_compliance=weights_data.get("guideline_compliance", 0.6),
    )
    return FitnessConfig(
        weights=weights,
        specificity_patterns=data.get("specificity_patterns", {}),
        calibration_keywords=data.get("calibration_keywords", {}),
        guideline_keywords=data.get("guideline_keywords", {}),
        deterministic_floors=data.get("deterministic_floors", {}),
        vague_phrases=data.get("vague_phrases", []),
        llm_prompts=data.get("llm_prompts", {}),
    )


def _parse_evolution_config(data: dict) -> EvolutionConfig:
    """Parse evolution configuration from YAML data."""
    pop = data.get("population", {})
    ops = data.get("operators", {})
    mut = ops.get("mutation", {})
    cx = ops.get("crossover", {})
    scenarios = data.get("scenarios", {})
    guidelines = data.get("guidelines", {})
    llm = data.get("llm", {})
    validation = data.get("validation", {})
    output = data.get("output", {})

    return EvolutionConfig(
        pop_size=pop.get("size", 16),
        n_generations=data.get("generations", 28),
        seed_profile_path=pop.get("seed_profile", ""),
        scenario_path=scenarios.get("path", ""),
        guidelines_path=guidelines.get("path", ""),
        output_dir=output.get("dir", "output"),
        llm_model=llm.get("model", "ollama/mistral:7b-instruct-v0.3-q5_K_M"),
        llm_temperature=llm.get("temperature", 0.7),
        llm_max_tokens=llm.get("max_tokens", 1024),
        llm_timeout=llm.get("timeout", 120),
        mutation_de_diff_weight=mut.get("de_diff_weight", 0.60),
        mutation_guideline_weight=mut.get("guideline_inject_weight", 0.25),
        mutation_prune_weight=mut.get("prune_weight", 0.15),
        crossover_prob=cx.get("probability", 0.3),
        validation_mode=validation.get("mode", "local"),
        validation_model=validation.get("claude_model", "anthropic/claude-sonnet-4-20250514"),
        validation_rounds=validation.get("pairwise_rounds", 3),
        scenario_rotation=scenarios.get("rotation", "round_robin"),
        save_every_gen=output.get("save_every_gen", True),
        consolidate_every=guidelines.get("consolidate_every", 5),
    )


def load_config(
    segment_schema_path: str,
    fitness_path: str,
    evolution_path: str,
) -> EvolverConfig:
    """Load and validate configuration from three YAML files.

    Args:
        segment_schema_path: Path to segment_schema.yaml
        fitness_path: Path to fitness.yaml
        evolution_path: Path to evolution.yaml

    Returns:
        Fully populated EvolverConfig.

    Raises:
        FileNotFoundError: If any config file is missing.
        KeyError: If required fields are missing.
    """
    for p in [segment_schema_path, fitness_path, evolution_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Config file not found: {p}")

    with open(segment_schema_path) as f:
        segment_data = yaml.safe_load(f)
    with open(fitness_path) as f:
        fitness_data = yaml.safe_load(f)
    with open(evolution_path) as f:
        evolution_data = yaml.safe_load(f)

    segments = _parse_segment_schema(segment_data)
    if not segments:
        raise ValueError("segment_schema.yaml must define at least one segment")

    fitness = _parse_fitness_config(fitness_data)
    evolution = _parse_evolution_config(evolution_data)

    return EvolverConfig(
        segments=segments,
        fitness=fitness,
        evolution=evolution,
    )
