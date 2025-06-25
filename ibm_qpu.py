from base import *

class StepCircuit():
    """
    Circuit structure around which the *current* QPU implementation takes.
    0 steps: only initial conditions and grid measurement.
    n steps: initial conditions, n CQLBM algorithm steps, and then grid measurement.
    """
    circuit: QuantumCircuit

    def __init__(self, lattice, num_steps, init_cond=None):
        if type(init_cond == None):
            self.circuit = CollisionlessInitialConditions(lattice).circuit
        else:
            self.circuit = init_cond
        for i in range(0, num_steps):
            self.circuit.compose(CQLBM(lattice).circuit, inplace=True)
        self.circuit.compose(GridMeasurement(lattice).circuit, inplace=True)
        #circuit.name = f"step {num_steps}"


class IBM_QPU_Runner():
    """
    Runs a 2D collisionless job with D_2Q_8 discretization on an IBM QPU.
    run: runs the job using the Qiskit Sampler primitive. Returns the job that has been run.
    visualize: takes the counts of the GridMeasurement data from the IBM QPU and turns it into a PyVista simulation.
    """

    lattice: CollisionlessLattice
    job_id: str
    dims: tuple
    service: QiskitRuntimeService

    def __init__(self, dims, token, vs=[4,4]):
        
        print(f"Initializing {dims[0]}x{dims[1]} runner... ", end="")
        self.service = QiskitRuntimeService(channel="ibm_quantum", 
                               token=token)
        
        self.dims = dims
        self.lattice = CollisionlessLattice(
            {
            "lattice": {"dim": {"x": dims[0], "y": dims[1]}, "velocities": {"x": vs[0], "y": vs[1]}},
            }
        )
        print("done.")

    def run(self, steps, shots=8192):

        print("Creating circuits... ", end="")

        step_qcs = [StepCircuit(self.lattice, i).circuit for i in range(steps+1)]

        backend = QiskitRuntimeService().least_busy(simulator=False, operational=True, min_num_qubits=25)
        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)

        qcs = [pass_manager.run(qc) for qc in step_qcs]

        sampler = Sampler(backend)
        
        print("done.")
        print(f"Sending {steps} step job to {backend} with {shots} shots per time-step... ", end="")

        job = sampler.run(qcs, shots=shots)
        self.job_id = job.job_id()

        print("done.")

        print(f"Job ID: {self.job_id}")

        return job

    
    def visualize(self, steps):

        print("Creating visualization... ", end="")

        job = self.service.job(self.job_id)
        results = job.result()
        counts_data = [list(results[i].data.values())[0].get_counts() for i in range(steps+1)]
 
        create_directory_and_parents(f"ibm-qpu-output\\collisionless-{self.dims[0]}x{self.dims[1]}-ibm-qpu")
        resultGen = CollisionlessResult(self.lattice, f"ibm-qpu-output\\collisionless-{self.dims[0]}x{self.dims[1]}-ibm-qpu")

        for i in range(steps+1):
            resultGen.save_timestep_counts(counts_data[i], i)
        resultGen.visualize_all_numpy_data()

        create_animation(f"ibm-qpu-output\\collisionless-{self.dims[0]}x{self.dims[1]}-ibm-qpu\\paraview", f"collisionless-{self.dims[0]}x{self.dims[1]}-ibm-qpu.gif")
        print("done.")

