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

    lattice: CollisionlessLattice | Lattice
    job_id: str
    dims: list | tuple
    service: QiskitRuntimeService
    backend: IBMBackend
    transpiled_circuits: list[QuantumCircuit]
    label: str
    error_mitigator: ErrorMitigator

    def __init__(
            self, 
            dims: list | tuple, 
            name: str, 
            vs: list | tuple = [4,4],
            readout_error_mitigation: bool = False,
            iterative_bayesian_unfolding: bool = False,
            zero_noise_extrapolation: bool = False,
            equalization: bool = False
        ) -> None:
        
        print(f"Initializing {dims[0]}x{dims[1]} runner... ", end="")
        self.service = QiskitRuntimeService(name=name)
        self.backend = QiskitRuntimeService().least_busy(simulator=False, operational=True, min_num_qubits=25)

        self.dims = dims
        self.lattice = Lattice(dims, vs=vs)

        self.error_mitigator = ErrorMitigator(
            self.lattice, 
            self.backend,
            self.service,
            readout_error_mitigation=readout_error_mitigation,
            iterative_bayesian_unfolding=iterative_bayesian_unfolding,
            zero_noise_extrapolation=zero_noise_extrapolation,
            equalization=equalization
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

        if self.error_mitigator.zero_noise_extrapolation:
            counts, self.label = self.error_mitigator.zne(shots, steps=steps)
            return counts
        else:

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
            shots: int = DEFAULT_SHOTS,
            job_id: str | None = None,
            counts: list[dict] | None = None
        ) -> str:
        """
        Takes the counts of the measurement data from the IBM QPU and turns it into a PyVista simulation.
        Returns the full gif file name.
        """
        
        print("Creating visualization... ")
        
        if self.error_mitigator.zero_noise_extrapolation == False:

            # IBU needs the transpiled circuits from the backend
            if (job_id != None):
                self.job_id = job_id
                self.backend = self.service.job(job_id).backend()
                step_qcs = [StepCircuit(self.lattice, i, collision=False).circuit for i in range(steps+1)]
                pass_manager = generate_preset_pass_manager(optimization_level=1, backend=self.backend)
                self.transpiled_circuits = [pass_manager.run(qc) for qc in step_qcs]

            job = self.service.job(self.job_id)
            results = job.result()
            raw_counts = [list(results[i].data.values())[0].get_counts() for i in range(steps+1)]

            counts, self.label = self.error_mitigator.mitigate(self.transpiled_circuits, shots, raw_counts)

        vis = super().visualize(counts, steps, shots=shots) # :)

        return vis
        

    @override
    def make(
            self, 
            steps: int, 
            shots: int = DEFAULT_SHOTS,
            init_cond: None | QuantumCircuit = None
        ) -> str:
        """
        Runs and visualizes the lattice.
        """
        if self.error_mitigator.zero_noise_extrapolation:
            counts = self.run(steps, shots=shots, init_cond=init_cond)
            vis = self.visualize(steps, shots=shots, counts=counts)
            return vis
        
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
            steps, shots=shots
        )
        return vis