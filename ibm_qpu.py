from base import *

class IBM_QPU_Runner(Runner):
    """
    Runs a 2D collisionless job with D_2Q_8 discretization on an IBM QPU.
    run: runs the job using the Qiskit Sampler primitive. Returns the job that has been run.
    visualize: takes the counts of the measurement data from the IBM QPU and turns it into a PyVista simulation.
    draw: draws the animation to the screen. 
    """

    stm_lattice: SpaceTimeLattice
    lattice: CollisionlessLattice
    job_id: str
    dims: tuple
    service: QiskitRuntimeService
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
            shots: int = 8192, 
            collision: bool = False, 
            init_cond: None | QuantumCircuit = None
        ):

        print("Creating and transpiling circuits... ", end="")

        if collision == True:
            step_qcs = [StepCircuit(self.stm_lattice, i, collision=True, init_cond=init_cond).circuit for i in range(steps+1)]
        else:
            step_qcs = [StepCircuit(self.lattice, i, collision=False, init_cond=init_cond).circuit for i in range(steps+1)]

        backend = QiskitRuntimeService().least_busy(simulator=False, operational=True, min_num_qubits=25)

        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
        qcs = [pass_manager.run(qc) for qc in step_qcs]

        options = SamplerOptions()
        options.dynamical_decoupling.enable = True
        # Turn on gate twirling. Requires qiskit_ibm_runtime 0.23.0 or later.
        options.twirling.enable_gates = True
        
        sampler = Sampler(backend, options=options)

        print("done.")
        print(f"Sending {steps} step job to {backend} with {shots} shots per time-step... ", end="")

        job = sampler.run(qcs, shots=shots)
        self.job_id = job.job_id()

        print("done.")

        print(f"Job ID: {self.job_id}")

        return job

    @override
    def visualize(
            self, 
            steps: int, 
            shots: int | None = None, 
            collision: bool = False
        ) -> str:
        if collision==False:
            self.label = f"collisionless-{self.dims[0]}x{self.dims[1]}-ibm-qpu"
        else:
            self.label = f"with-collision-{self.dims[0]}x{self.dims[1]}-ibm-qpu"

        print("Creating visualization... ", end="")

        job = self.service.job(self.job_id)
        results = job.result()
        counts_data = [list(results[i].data.values())[0].get_counts() for i in range(steps+1)]

        try:
            rmtree(f"ibm-qpu-output\\{self.label}")
        except OSError:
            pass

        create_directory_and_parents(f"ibm-qpu-output\\{self.label}")
        if collision == False:
            resultGen = CollisionlessResult(self.lattice, f"ibm-qpu-output\\{self.label}")
        else:
            resultGen = SpaceTimeResult(self.stm_lattice, f"ibm-qpu-output\\{self.label}")

        for i in range(steps+1):
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
            shots: int = 8192, 
            init_cond: None | QuantumCircuit = None
        ) -> str:
        
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
        vis = self.visualize(steps, shots=shots)
        return vis