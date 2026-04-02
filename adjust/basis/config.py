"""Configuration for the BASIS pipeline."""

from dataclasses import dataclass, field
from typing import List, Any


@dataclass
class DatasetConfig:
    """Per-dataset preprocessing options."""
    path: str = ""
    log_transform: bool = False
    label: str = ""  # auto-assigned if empty


@dataclass
class BASISConfig:
    """All pipeline parameters in one place.

    datasets[0] is always the reference. datasets[1:] are targets.
    """

    # Datasets (first = reference, rest = targets)
    datasets: List[DatasetConfig] = field(default_factory=list)

    # Output
    output_dir: str = ""
    viz: bool = True
    meta_prefix: str = "meta_"
    keep_shared_only: bool = True

    # Advanced Merge Strategies
    merge_order: Any = None
    auto_merge: bool = False
    progressive: bool = False

    # Deduplication
    dedup_threshold: float = 0.999
    corr_ceiling: float = 0.99

    # Edge computation
    d_threshold: float = 0.5
    w_floor: float = 0.25
    top_k_edges: int = 200

    # Resolution optimization
    gp_n_calls: int = 25
    res_range_min: float = 1.0
    res_range_max: float = 200.0

    # Consensus Leiden
    n_runs: int = 20
    consensus_threshold: float = 0.7

    # Dictionary construction
    greedy_merge_threshold: float = 0.7
    ghost_gene_floor: float = 0.05

    # OT alignment
    ot_epsilon: float = 0.01
    ot_tau: float = 0.1

    def dict_config(self):
        """Extract dictionary-building parameters as a dict."""
        return {
            "dedup_threshold": self.dedup_threshold,
            "d_threshold": self.d_threshold,
            "w_floor": self.w_floor,
            "top_k_edges": self.top_k_edges,
            "corr_ceiling": self.corr_ceiling,
            "gp_n_calls": self.gp_n_calls,
            "res_range_min": self.res_range_min,
            "res_range_max": self.res_range_max,
            "n_runs": self.n_runs,
            "consensus_threshold": self.consensus_threshold,
            "greedy_merge_threshold": self.greedy_merge_threshold,
            "ghost_gene_floor": self.ghost_gene_floor,
        }

    @property
    def ref(self):
        return self.datasets[0] if self.datasets else DatasetConfig()

    @property
    def targets(self):
        return self.datasets[1:] if len(self.datasets) > 1 else []
