from dataclasses import dataclass
from typing import List, Literal

from util.hparams import HyperParams


@dataclass
class AlphaEditHyperParams(HyperParams):
    # Method
    model_name: str
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    nullspace_threshold: float
    L2: float
    numerical_stability: str = "none"
    stability_spectral_multiplier: float = 0.0
    stability_condition_number: float = 0.0
    stability_fro_drift_ratio: float = 0.0
    knowledge_conflict: str = "none"
    conflict_loss_weight: float = 0.0
    conflict_margin: float = 0.0
    analysis_top_singular_values: int = 8
    # Delta-level Frobenius clipping (AlphaEditDeltaFro)
    delta_fro_tau: float = 0.0        # absolute threshold; 0 = use ratio
    delta_fro_ratio: float = 0.0      # tau_F = ratio * ||W0||_F; 0 = disabled
    # Delta-level Spectral clipping (AlphaEditDeltaSpectral)
    delta_spectral_tau: float = 0.0   # absolute threshold; 0 = disabled
