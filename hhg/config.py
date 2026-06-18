from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np

BoundaryCondition = Literal["open", "periodic"]
PhaseDistanceMode = Literal["legacy", "bond_length"]
SolverName = Literal["quspin", "density_matrix", "crank_nicolson"]

@dataclass(frozen=True)
class SSHModelConfig:
    """
    Congiguration for the SSH model.

    Attributes:
        num_lattice_points: Number of lattice points in the system. Must be positive and even.
        lattice_constant: Lattice constant in atomic units. Must be positive.
        delta: Alternating shift of the atoms causing the dimerization.
        boundary_condition: Boundary condition for the system. Must be either "open" or "periodic".
        phase_distance_mode: Mode for calculating the phase distance. Must be either "legacy" or "bond_length".
        n_particles_per_basis: Number of particles per basis. Must be positive.
    """
    num_lattice_points: int = 100
    lattice_constant: float = 2.0
    delta: float = 0.15
    boundary_condition: BoundaryCondition = "open"
    phase_distance_mode: PhaseDistanceMode = "legacy"
    num_particles_per_basis: int = 1

    def __post_init__(self):
        if self.num_lattice_points <= 0:
            raise ValueError("num lattice points must be positive.")
        
        if self.num_lattice_points % 2 != 0:
            raise ValueError("num lattice points must be even.")
        
        if self.lattice_constant <= 0:
            raise ValueError("lattice constant must be positive.")

        if self.boundary_condition not in ("open", "periodic"):
            raise ValueError("boundary condition must be either 'open' or 'periodic'.")
        
        if self.phase_distance_mode not in ("legacy", "bond_length"):
            raise ValueError("phase distance mode must be either 'legacy' or 'bond_length'.")
        
        if self.num_particles_per_basis <= 0:
            raise ValueError("number of particles per basis must be positive.")
        
    @property
    def num_occupied_states(self):
        return self.num_lattice_points // 2

@dataclass(frozen=True)
class VectorPotentialConfig:
    """
    Configuration for the laser pulse.

    Attributes:
        num_cycles: Number of cycles in the laser pulse. Must be positive.
        frequency: Frequency of the laser pulse in atomic units. Must be positive.
        amplitude: Amplitude of the laser pulse. Must be non-negative.

    The laser pulse is defined as,
        A(t) = A_0 * sin(omega * t) * sin^2(frequency * t / (2 * num_cycles))
    """
    num_cycles: int = 5
    frequency: float = 0.0075
    amplitude: float = 0.2

    def __post_init__(self):
        if self.num_cycles <= 0:
            raise ValueError("number of cycles must be positive.")
        
        if self.frequency <= 0:
            raise ValueError("frequency must be positive.")
        
        if self.amplitude < 0:
            raise ValueError("amplitude must be non-negative.")
        
    @property
    def final_time(self):
        return 2 * np.pi * self.num_cycles / self.frequency
    
@dataclass(frozen=True)
class TimeGridConfig:
    """
    Configuration for the time grid.

    Attributes:
        delta_t: Time step for the time grid. Must be positive.
        t_start: Start time for the time grid.
        t_stop: Stop time for the time grid. If None, the time grid will be infinite.
        endpoint: Whether to include the stop time in the time grid.
    """

    delta_t: float = 1.0
    t_start: float = 0.0
    t_stop: float | None = None
    endpoint: bool = False

    def __post_init__(self):
        if self.delta_t <= 0:
            raise ValueError("delta t must be positive.")
        
        if self.t_stop is not None and self.t_stop <= self.t_start:
            raise ValueError("t stop must be greater than t start.")

@dataclass(frozen=True)
class SpectrumConfig:
    """
    Configuration for the spectrum calculation.

    Attributes:
         use_pulse_mask: Whether to apply a pulse mask to the time series before calculating the spectrum.
         max_harmonic_order: Maximum harmonic order to calculate in the spectrum. Must be positive.
    """
    use_pulse_mask: bool = True
    max_harmonic_order: int = 100

    def __post_init__(self):
        if self.max_harmonic_order <= 0:
            raise ValueError("max harmonic order must be positive.")

@dataclass(frozen=True)
class OutputConfig:
    """
    Configuration for the output of the simulation.
    
    Attributes:
        root_dir: Root directory for the output. Must be a valid path.
        run_name: Name of the run. Used to create a subdirectory in the root directory for the output.
    """

    root_dir: Path = Path("results")
    run_name: str = "ssh_hhg"

    def __post_init__(self):
        if not self.root_dir.exists():
            self.root_dir.mkdir(parents=True, exist_ok=True)
        
    @property
    def output_dir(self):
        return self.root_dir / self.run_name
    
@dataclass(frozen=True)
class ExperimentConfig:
    """
    Configuration for the entire experiment, including the SSH model, laser pulse, time grid, spectrum calculation, and output.
    Also includes the choice of solver for the time evolution, which can be either "quspin", "density_matrix", or "crank_nicolson".

    Attributes:
        model: Configuration for the SSH model.
        vector_potential: Configuration for the laser pulse.
        time: Configuration for the time grid.
        spectrum: Configuration for the spectrum calculation.
        output: Configuration for the output of the simulation.
        solver: Choice of solver for the time evolution. Must be either "quspin", "density_matrix", or "crank_nicolson".
    """

    model: SSHModelConfig
    vector_potential: VectorPotentialConfig
    time: TimeGridConfig
    spectrum: SpectrumConfig
    output: OutputConfig
    solver: SolverName = "quspin"

    def __post_init__(self):
        if self.solver not in ("quspin", "density_matrix", "crank_nicolson"):
            raise ValueError("solver must be either 'quspin', 'density_matrix', or 'crank_nicolson'.")