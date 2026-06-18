from hhg.io_config import load_experiment_config

config = load_experiment_config("configs/quspin_operator.yaml")

print(config)
print("Model Lattice Size:", config.model.num_lattice_points)
print("Delta", config.model.delta)
print("Vector Potential Amplitude:", config.vector_potential.amplitude)
print("Vector Potential Frequency:", config.vector_potential.frequency)
print("Time Grid Start:", config.time.t_start)
