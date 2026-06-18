from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import (SSHModelConfig,
                     VectorPotentialConfig, 
                     TimeGridConfig, 
                     SpectrumConfig, 
                     OutputConfig, 
                     ExperimentConfig)


def load_yaml_file(file_path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.
    Parameters
    ----------
    file_path : str or Path
        The path to the YAML file to load.
    
    Returns
    -------
    dict
        The contents of the YAML file as a dictionary.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")    
    
    with open(file_path, "r") as f:
        try:
            data = yaml.safe_load(f)
        
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
    
    if data is None:
        raise ValueError(f"YAML file is empty: {file_path}")
    
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a dictionary at the top level: {file_path}")
    
    return data

def load_experiment_config(file_path: str | Path) -> ExperimentConfig:
    """
    Load an experiment configuration from a YAML file and return an ExperimentConfig object.
    """

    data = load_yaml_file(file_path)

    required_keys = ["model", "vector_potential", "time", "spectrum", "output", "solver"]

    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in YAML file: {file_path}")

    model_config = SSHModelConfig(**data["model"])
    vector_potential_config = VectorPotentialConfig(**data["vector_potential"])
    time_config = TimeGridConfig(**data["time"])
    spectrum_config = SpectrumConfig(**data["spectrum"])

    output_data = data["output"].copy()
    output_data["root_dir"] = Path(output_data["root_dir"])

    output_config = OutputConfig(**output_data)

    return ExperimentConfig(model=model_config,
                            vector_potential=vector_potential_config,
                            time=time_config,
                            spectrum=spectrum_config,
                            output=output_config,
                            solver=data["solver"])






