from hhg.config import (
    SSHModelConfig,
    VectorPotentialConfig,
    TimeGridConfig,
    SpectrumConfig,
    OutputConfig,
    ExperimentConfig,

)

config = ExperimentConfig(
    model=SSHModelConfig(),
    vector_potential=VectorPotentialConfig(),
    time=TimeGridConfig(),
    spectrum=SpectrumConfig(),
    output=OutputConfig(),
    solver="quspin",
)

print(config)
print("Number of occupied states:", config.model.num_occupied_states)
print("Laser final time:", config.vector_potential.final_time)
print("Output directory:", config.output.output_dir)