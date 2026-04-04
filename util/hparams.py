import json
from dataclasses import dataclass


@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """

    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)

        if "hparams" in data and isinstance(data["hparams"], dict):
            data = data["hparams"]

        return cls(**data)
