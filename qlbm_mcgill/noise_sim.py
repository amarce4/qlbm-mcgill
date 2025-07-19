from base import *

class Noise_Simulation2D(Runner):
    """
    Performs a noisy 2D QLBM simulation without reinitialization (i.e., significantly less optimized).
    Scales in O(n^2) for n steps compared to Simulation2D's O(n).
    Attributes:
        single_depolarizing_prob: float, single qubit gate error probability
        double_depolarizing_prob: float, double qubit gate error probability
        noise_model: NoiseModel
        lattice: CollisionlessLattice
        dims: tuple | list
        label: str, name of the file without the extension
    """
     
    single_depolarizing_prob: float
    double_depolarizing_prob: float
    noise_model: NoiseModel
    lattice: CollisionlessLattice
    dims: tuple | list
    label: str

    def __init__(
            self, 
            single_prob: float, 
            double_prob: float, 
            dims: tuple | list, 
            vs: tuple | list = [4,4]
        ) -> None:
          
        self.dims = dims
        self.lattice = CollisionlessLattice(
            {
            "lattice": {"dim": {"x": dims[0], "y": dims[1]}, "velocities": {"x": vs[0], "y": vs[1]}},
            }
        )

        self.single_depolarizing_prob = single_prob
        self.double_depolarizing_prob = double_prob

        self.noise_model = NoiseModel()
        self.noise_model.add_all_qubit_quantum_error(
            depolarizing_error(double_prob, 2), ["cx"]
        )
        self.noise_model.add_all_qubit_quantum_error(
            depolarizing_error(single_prob, 1), ["u", "u3", "p", "h", "measure"]
        )

    @override
    def run(
        self, 
        steps: int, 
        shots: int = DEFAULT_SHOTS
        ) -> list:

        step_qcs = [StepCircuit(self.lattice, i).circuit.decompose() for i in range(steps+1)]
        
        noisy_sampler = SimSampler(
            options=dict(backend_options=dict(noise_model=self.noise_model))
        )
        # The circuit needs to be transpiled to the AerSimulator target
        pass_manager = generate_preset_pass_manager(3, AerSimulator())
        qcs = [pass_manager.run(qc) for qc in step_qcs]
        job = noisy_sampler.run(qcs, shots=shots)
        result = job.result()
        
        counts = [list(result[i].data.values())[0].get_counts() for i in range(steps+1)]
        return counts

    @override
    def visualize(
        self, 
        counts: list,
        steps: int, 
        shots: int = DEFAULT_SHOTS
        ) -> str:

        self.label = f"noisy-collisionless-simulation-{self.dims[0]}x{self.dims[1]}_{self.single_depolarizing_prob}-single-{self.double_depolarizing_prob}-double"

        return super().visualize(counts, steps, shots=shots) # :)

    @override
    def make(
        self, 
        steps: int, 
        shots: int = DEFAULT_SHOTS
        ) -> str:
        print(f"Running {self.dims[0]}x{self.dims[1]} simulation with {self.single_depolarizing_prob} single and {self.double_depolarizing_prob} double gate error probabilities...")
        counts = self.run(steps, shots=shots)
        print(f"Creating visualization...")
        vis = self.visualize(counts, steps, shots=shots)
        print("Done.")
        return vis 
