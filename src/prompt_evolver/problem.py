"""pymoo Problem subclass for prompt evolution."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
from pymoo.core.problem import ElementwiseProblem

if TYPE_CHECKING:
    from prompt_evolver.config import EvolverConfig
    from prompt_evolver.fitness import FitnessEvaluator
    from prompt_evolver.genome import Genome


class PromptEvolutionProblem(ElementwiseProblem):
    """Multi-objective prompt evolution problem for pymoo.

    n_var = 1 (single object-typed variable: the Genome)
    n_obj = 6 (the 6 fitness components, negated for minimization)
    n_ieq_constr = 1 (hard floor constraint)
    """

    def __init__(
        self,
        fitness_evaluator: FitnessEvaluator,
        scenarios: list[dict],
        config: EvolverConfig,
    ):
        super().__init__(
            n_var=1,
            n_obj=6,
            n_ieq_constr=1,
        )
        self.fitness_evaluator = fitness_evaluator
        self.scenarios = scenarios
        self.config = config
        self._scenario_idx = 0

    def _evaluate(self, x, out, *args, **kwargs):
        genome: Genome = x[0]

        scenario = self._select_scenario()
        result = self.fitness_evaluator.evaluate(genome, scenario)

        # pymoo minimizes — negate so higher fitness = lower objective value
        out["F"] = result.as_weighted_array(self.config.fitness.weights)

        # Hard constraint: deterministic floors
        if result.passes_floors(self.config.fitness.deterministic_floors):
            out["G"] = np.array([-1.0])  # feasible
        else:
            out["G"] = np.array([1.0])  # infeasible

    def _select_scenario(self) -> dict:
        """Select next scenario via round-robin or random."""
        if not self.scenarios:
            return {"description": "", "expected_focus": []}

        if self.config.evolution.scenario_rotation == "round_robin":
            scenario = self.scenarios[self._scenario_idx % len(self.scenarios)]
            self._scenario_idx += 1
            return scenario
        else:
            return random.choice(self.scenarios)
