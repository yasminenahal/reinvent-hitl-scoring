from dataclasses import dataclass

import numpy as np
from typing import Dict, List

from reinvent_scoring.scoring.component_parameters import ComponentParameters

stan = False

@dataclass
class ComponentSummary:
    total_score: np.array
    parameters: ComponentParameters
    raw_score: np.ndarray = None
    prob_dists: np.ndarray = None
    dist_bioactivity_0: np.array = None
    dist_bioactivity_1: np.array = None


class FinalSummary:
    def __init__(self, total_score: np.array, scored_smiles: List[str], valid_idxs: List[int],
                 scaffold_log_summary: List[ComponentSummary]):
        self.total_score = total_score
        self.scored_smiles = scored_smiles
        self.valid_idxs = valid_idxs
        score = [LoggableComponent(c.parameters.component_type, c.parameters.name, c.total_score) for c in scaffold_log_summary]
        raw_score = [LoggableComponent(c.parameters.component_type, f'raw_{c.parameters.name}', c.raw_score) for c in
                     scaffold_log_summary if c.raw_score is not None]
        score.extend(raw_score)
        if stan:
            prob0 = [LoggableComponent(c.parameters.component_type, f'dist_{c.parameters.name}_0', c.prob_dists[:,0]) for c in
                     scaffold_log_summary if c.prob_dists is not None]
            prob1 = [LoggableComponent(c.parameters.component_type, f'dist_{c.parameters.name}_1', c.prob_dists[:,1]) for c in
                     scaffold_log_summary if c.prob_dists is not None]        
            score.extend(prob0)
            score.extend(prob1)
        
        self.scaffold_log: List[ComponentSummary] = scaffold_log_summary
        self.profile: List[LoggableComponent] = score



@dataclass
class LoggableComponent:
    component_type: str
    name: str
    score: np.array
