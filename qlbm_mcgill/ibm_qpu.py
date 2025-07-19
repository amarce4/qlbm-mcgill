from base import *
from error_mitigator import ErrorMitigator

class IBM_QPU_Runner(Runner):
    """
    Runs a 2D QLBM on an IBM QPU.
    Attributes:
        stm_lattice: SpaceTimeLattice
        lattice: CollisionlessLattice
        job_id: str
        dims: tuple
        service: QiskitRuntimeService
        label: str, name of the file without extension
    """

    stm_lattice: SpaceTimeLattice
    lattice: CollisionlessLattice | Lattice
    job_id: str
    dims: list | tuple
    service: QiskitRuntimeService
    backend: IBMBackend
    transpiled_circuits: list[QuantumCircuit]
    label: str

    def __init__(
            self, 
            dims: list | tuple, 
            name: str, 
            vs: list | tuple = [4,4]
        ) -> None:
        
        print(f"Initializing {dims[0]}x{dims[1]} runner... ", end="")
        self.service = QiskitRuntimeService(name=name)
        self.backend = QiskitRuntimeService().least_busy(simulator=False, operational=True, min_num_qubits=25)

        self.dims = dims
        self.lattice = Lattice(dims, vs)
        self.stm_lattice = SpaceTimeLattice(
            num_timesteps=1,
            lattice_data={
                "lattice": {"dim": {"x": dims[0], "y": dims[1]}, "velocities": {"x": 2, "y": 2}},
                "geometry": [],
            },
        )
        print("done.")

    @override
    def run(
            self, 
            steps: int, 
            shots: int = DEFAULT_SHOTS, 
            init_cond: None | QuantumCircuit = None
        ):
        """
        Runs the job using the Qiskit Sampler primitive. Returns the job that has been run.
        """
        print("Creating and transpiling circuits... ", end="")

        step_qcs = [StepCircuit(self.lattice, i, collision=False, init_cond=init_cond).circuit for i in range(steps+1)]

        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=self.backend)
        self.transpiled_circuits = [pass_manager.run(qc) for qc in step_qcs]

        options = SamplerOptions()
        options.dynamical_decoupling.enable = True
        # Turn on gate twirling. Requires qiskit_ibm_runtime 0.23.0 or later.
        options.twirling.enable_gates = True
        
        sampler = Sampler(self.backend, options=options)

        print("done.")
        print(f"Sending {steps} step job to {self.backend} with {shots} shots per time-step... ", end="")

        job = sampler.run(self.transpiled_circuits, shots=shots)
        self.job_id = job.job_id()

        print("done.")

        print(f"Job ID: {self.job_id}")

        return job

    @override
    def visualize(
            self, 
            steps: int, 
            shots: int,
            job_id: str | None = None,
            readout_error_mitigation: bool = False,
            iterative_bayesian_unfolding: bool = False
        ) -> str:
        """
        Takes the counts of the measurement data from the IBM QPU and turns it into a PyVista simulation.
        Returns the full gif file name.
        """
        
        print("Creating visualization... ")
        
        # IBU needs the transpiled circuits from the backend
        if (job_id != None):
            self.job_id = job_id
            step_qcs = [StepCircuit(self.lattice, i, collision=False).circuit for i in range(steps+1)]
            pass_manager = generate_preset_pass_manager(optimization_level=1, backend=self.backend)
            self.transpiled_circuits = [pass_manager.run(qc) for qc in step_qcs]

        job = self.service.job(self.job_id)
        results = job.result()
        raw_counts = [list(results[i].data.values())[0].get_counts() for i in range(steps+1)]

        error_mitigator = ErrorMitigator(
            self.lattice, 
            self.backend,
            readout_error_mitigation=readout_error_mitigation,
            iterative_bayesian_unfolding=iterative_bayesian_unfolding
        )

        counts, self.label = error_mitigator.mitigate(self.transpiled_circuits, shots, raw_counts)

        vis = super().visualize(counts, steps, shots=shots) # :)

        return vis
        

    @override
    def make(
            self, 
            steps: int, 
            shots: int = DEFAULT_SHOTS, 
            readout_error_mitigation: bool = False,
            iterative_bayesian_unfolding: bool = False,
            init_cond: None | QuantumCircuit = None
        ) -> str:
        """
        Runs and visualizes the lattice.
        """
        job = self.run(steps, shots=shots, init_cond=init_cond)
        
        start = int(time.time())
        print("Waiting for IBM QPU data...")
        time.sleep(1)
        print()
        while (job.status() != "DONE"):
            time.sleep(0.9)
            diff = int(time.time()) - start
            print("\r", end="")
            print(f"Time elapsed: {int(diff/60)} minute(s) and {diff % 60} second(s).", end="")
        print(f"\nData received. Workload: {int(job.usage())} seconds.")
        time.sleep(2) # make them wait for it.
        vis = self.visualize(
            steps, shots=shots, 
            readout_error_mitigation=readout_error_mitigation,
            iterative_bayesian_unfolding=iterative_bayesian_unfolding
        )
        return vis