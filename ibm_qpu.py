from base import *

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
    lattice: CollisionlessLattice
    job_id: str
    dims: tuple
    service: QiskitRuntimeService
    backend: IBMBackend
    label: str

    def __init__(
            self, 
            dims: list | tuple, 
            name: str, 
            vs: list | tuple = [4,4]
        ) -> None:
        
        print(f"Initializing {dims[0]}x{dims[1]} runner... ", end="")
        self.service = QiskitRuntimeService(name=name)
        
        self.dims = dims
        self.lattice = CollisionlessLattice(
            {
            "lattice": {"dim": {"x": dims[0], "y": dims[1]}, "velocities": {"x": vs[0], "y": vs[1]}},
            }
        )
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

        self.backend = QiskitRuntimeService().least_busy(simulator=False, operational=True, min_num_qubits=25)

        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=self.backend)
        qcs = [pass_manager.run(qc) for qc in step_qcs]

        options = SamplerOptions()
        options.dynamical_decoupling.enable = True
        # Turn on gate twirling. Requires qiskit_ibm_runtime 0.23.0 or later.
        options.twirling.enable_gates = True
        
        sampler = Sampler(self.backend, options=options)

        print("done.")
        print(f"Sending {steps} step job to {self.backend} with {shots} shots per time-step... ", end="")

        job = sampler.run(qcs, shots=shots)
        self.job_id = job.job_id()

        print("done.")

        print(f"Job ID: {self.job_id}")

        return job

    @override
    def visualize(
            self, 
            steps: int, 
            readout_error_mitigation: bool = False,
            shots: int | None = None
        ) -> str:
        """
        Takes the counts of the measurement data from the IBM QPU and turns it into a PyVista simulation.
        Returns the full gif file name.
        """
        
        print("Creating visualization... ")

        job = self.service.job(self.job_id)
        results = job.result()
        counts_data = [list(results[i].data.values())[0].get_counts() for i in range(steps+1)]

        
        if readout_error_mitigation:
            print("Mitigating readout error...", end="")

            self.label = f"mitigated-collisionless-{self.dims[0]}x{self.dims[1]}-ibm-qpu"

            measured_qubits = StepCircuit(self.lattice, 0).grid_qubits
            exp = LocalReadoutError(measured_qubits)

            exp.analysis.set_options(plot=True)
            result = exp.run(self.backend)
            mitigator = result.analysis_results("Local Readout Mitigator", dataframe=True).iloc[0].value
            mitigated_quasi_probs = [mitigator.quasi_probabilities(counts) for counts in counts_data]

            mitigated_probs = [(prob.nearest_probability_distribution().binary_probabilities()) for prob in mitigated_quasi_probs]

            mitigated_counts = [{label: int(prob * shots) for label, prob in mitigated_prob.items()} for mitigated_prob in mitigated_probs]
        else:
            self.label = f"collisionless-{self.dims[0]}x{self.dims[1]}-ibm-qpu"


        try:
            rmtree(f"ibm-qpu-output\\{self.label}")
        except OSError:
            pass

        rmdir_rf(f"ibm-qpu-output\\{self.label}")
        create_directory_and_parents(f"ibm-qpu-output\\{self.label}")
        
        resultGen = CollisionlessResult(self.lattice, f"ibm-qpu-output\\{self.label}")
        
        for i in range(steps+1):
            if readout_error_mitigation:
                resultGen.save_timestep_counts(mitigated_counts[i], i)
            else:    
                resultGen.save_timestep_counts(counts_data[i], i)
        resultGen.visualize_all_numpy_data()

        if (type(shots) == None):
            create_animation(f"ibm-qpu-output\\{self.label}\\paraview", f"{self.label}.gif")
        elif (type(shots) == int):
            create_animation(f"ibm-qpu-output\\{self.label}\\paraview", f"{self.label}_{shots}_shots.gif")
        print("done.")
        print(f"Animation saved as ''{self.label}_{shots}_shots.gif''.")
        return f"{self.label}_{shots}_shots.gif"

    @override
    def make(
            self, 
            steps: int, 
            shots: int = DEFAULT_SHOTS, 
            readout_error_mitigation: bool = False,
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
        vis = self.visualize(steps, shots=shots, readout_error_mitigation=readout_error_mitigation)
        return vis