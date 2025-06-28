from base import *

class StepCircuit():
    """
    Circuit structure around which the *current* QPU implementation takes.
    0 steps: only initial conditions and grid measurement.
    n steps: initial conditions, n CQLBM algorithm steps, and then grid measurement.
    """
    circuit: QuantumCircuit

    def __init__(self, 
                 lattice: CollisionlessLattice | SpaceTimeLattice, 
                 num_steps: int, 
                 init_cond: None | QuantumCircuit = None, 
                 collision: bool = False):
        if collision == False:
            if type(init_cond == None):
                self.circuit = CollisionlessInitialConditions(lattice).circuit
            else:
                self.circuit = init_cond
            for i in range(0, num_steps):
                self.circuit.compose(CQLBM(lattice).circuit, inplace=True)
                self.circuit.reset([-3,-4,0,1])
            self.circuit.compose(GridMeasurement(lattice).circuit, inplace=True)
        else:
            if type(init_cond == None):
                self.circuit = PointWiseSpaceTimeInitialConditions(lattice, grid_data=[((1, 5), (True, True, True, True))]).circuit
            else:
                self.circuit = init_cond
            for i in range(0, num_steps):
                self.circuit.compose(SpaceTimeQLBM(lattice).circuit, inplace=True)
            self.circuit.compose(SpaceTimeGridVelocityMeasurement(lattice).circuit, inplace=True)
        self.circuit = remove_idle_wires(self.circuit)
        


class IBM_QPU_Runner():
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
            token: str, 
            vs: list | tuple = [4,4]):
        
        print(f"Initializing {dims[0]}x{dims[1]} runner... ", end="")
        self.service = QiskitRuntimeService(channel="ibm_quantum", 
                               token=token)
        
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

    def run(
            self, 
            steps: int, 
            shots: int = 8192, 
            collision: bool = False, 
            init_cond: None | QuantumCircuit = None):

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
    
    def visualize(
            self, 
            steps: int, 
            shots: int | None = None, 
            collision: bool = False):
        if collision==False:
            self.label = "collisionless"
        else:
            self.label = "with-collision"

        print("Creating visualization... ", end="")

        job = self.service.job(self.job_id)
        results = job.result()
        counts_data = [list(results[i].data.values())[0].get_counts() for i in range(steps+1)]
 
        create_directory_and_parents(f"ibm-qpu-output\\{self.label}-{self.dims[0]}x{self.dims[1]}-ibm-qpu")
        if collision == False:
            resultGen = CollisionlessResult(self.lattice, f"ibm-qpu-output\\{self.label}-{self.dims[0]}x{self.dims[1]}-ibm-qpu")
        else:
            resultGen = SpaceTimeResult(self.stm_lattice, f"ibm-qpu-output\\{self.label}-{self.dims[0]}x{self.dims[1]}-ibm-qpu")

        for i in range(steps+1):
            resultGen.save_timestep_counts(counts_data[i], i)
        resultGen.visualize_all_numpy_data()

        if (type(shots) == None):
            create_animation(f"ibm-qpu-output\\{self.label}-{self.dims[0]}x{self.dims[1]}-ibm-qpu\\paraview", f"{self.label}-{self.dims[0]}x{self.dims[1]}-ibm-qpu.gif")
        elif (type(shots) == int):
            create_animation(f"ibm-qpu-output\\{self.label}-{self.dims[0]}x{self.dims[1]}-ibm-qpu\\paraview", f"{self.label}-{self.dims[0]}x{self.dims[1]}-ibm-qpu_{shots}_shots.gif")
        print("done.")
        print(f"Animation saved as ''{self.label}-{self.dims[0]}x{self.dims[1]}-ibm-qpu_{shots}_shots.gif''.")

    def make(
            self, 
            steps: int, 
            shots: int = 8192, 
            init_cond: None | QuantumCircuit = None):
        
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
        self.visualize(steps, shots=shots)