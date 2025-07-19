from base import *

class REMTable:
    """
    A lookup table for Readout Error Mitigation (REM) experiment data, to save on QPU time.
    """
    service: QiskitRuntimeService

    def __init__(self, service):
        self.service = service

    def enter(
        self, 
        dims: list | tuple, 
        job_id: str
        ):
        """
        Enter a REM experiment into the table by dumping its contents into a .json file.
        """
        job = self.service.job(job_id)
        with open(f"rem-table/{job.backend().name}_{dims[0]}x{dims[1]}.json", "w") as file:
            json.dump((job.result(), job_id), file, cls=RuntimeEncoder)

    def load(
        self, 
        dims: list | tuple, 
        backend: IBMBackend,
        exp: CorrelatedReadoutError
        ):
        """
        Load REM experiment results from the corresponding .json file.
        """
        with open(f"rem-table/{backend.name}_{dims[0]}x{dims[1]}.json", "r") as file:
            raw, job_id = json.load(file, cls=RuntimeDecoder)
            data = ExperimentData(experiment=exp)
            data._add_result_data(raw, job_id)
            result = exp.analysis.run(data)

        return result

def get_measured_qubits(circuit):
    """
    Gets the measured qubits of a circuit.
    AI slop, needs improvement.
    """
    measured_qubits = []
    for instruction, qargs, cargs in circuit.data:
        if instruction.name == 'measure':
            bits = [circuit.find_bit(qarg)[0] for qarg in qargs]
            measured_qubits.append(bits[0])
    return measured_qubits
    

class ErrorMitigator:
    """
    Contains all methods that mitigate error via post-processing.
    Currently implements Readout Error Mitigation (REM) and Iterative Bayesian Unfolding (IBU).
    """
    class ReadoutError:
        """
        Retrieves and/or enters results from the REM table based on the boolean "use_table".
        """
        result: ExperimentData
        
        def __init__(
            self,
            lattice: Lattice,
            backend: IBMBackend,
            service: QiskitRuntimeService,
            use_table: bool,
            ) -> None:

            dims = lattice.dims
            measured_qubits = StepCircuit(lattice, 1).grid_qubits
            exp = CorrelatedReadoutError(measured_qubits)
            
            table = REMTable(service)
            if use_table and path.exists(f"rem-table/{backend.name}_{dims[0]}x{dims[1]}.json"):
                    result = table.load(dims, backend, exp)
            else:
                result = exp.run(backend)
                table.enter(dims, result.job_ids[0])
            
            self.result = result

    dims: list | tuple
    lattice: CollisionlessLattice | Lattice
    backend: IBMBackend
    service: QiskitRuntimeService

    def __init__(
            self,
            lattice: CollisionlessLattice | Lattice,
            backend: IBMBackend,
            service: QiskitRuntimeService,
            readout_error_mitigation: bool = False,
            iterative_bayesian_unfolding: bool = False
            ) -> None:
        self.lattice = lattice
        self.dims = lattice.dims
        self.backend = backend
        self.service = service
        self.readout_error_mitigation = readout_error_mitigation
        self.iterative_bayesian_unfolding = iterative_bayesian_unfolding

    
    def rem(
            self,
            shots: int,
            counts: list,
            use_table: bool = True
        ) -> list:
            
        print("Mitigating readout error...")

        result = ErrorMitigator.ReadoutError(self.lattice, self.backend, self.service, use_table).result

        mitigator = result.analysis_results("Correlated Readout Mitigator", dataframe=True).iloc[0].value
        mitigated_quasi_probs = [mitigator.quasi_probabilities(count) for count in counts]

        mitigated_probs = [(prob.nearest_probability_distribution().binary_probabilities()) for prob in mitigated_quasi_probs]

        mitigated_counts = [{label: int(prob * shots) for label, prob in mitigated_prob.items()} for mitigated_prob in mitigated_probs]
        
        return mitigated_counts
        
    def ibu(
            self,
            qcs: list[QuantumCircuit],
            shots: int,
            counts: list,
        ) -> list:
        """
        (NOT MY CODE)
        Modified based on PyIBU tutorial.ipynb
        Credit: https://github.com/sidsrinivasan/PyIBU
        """
        print("Performing Iterative Bayesian Unfolding...")

        measured_qubits_list = [get_measured_qubits(qc) for qc in qcs]

        params = {
            "exp_name": "qlbm",
            "method": "full",  # options: "full", "reduced"
            "library": "jax",  # options: "tensorflow" (for "full" only) or "jax"
            "num_qubits": len(measured_qubits_list[0]),
            "max_iters": 100,
            "tol": 1e-4,
            "use_log": False,  # options: True or False
            "verbose": False,
            "init": "unif",  # options: "unif" or "unif_obs" or "obs"
            "smoothing": 1e-8
        }
        if params["library"] == 'tensorflow':
            params.update({
                "max_iters": tf.constant(params["max_iters"]),
                "eager_run": True
            })
            tf.config.run_functions_eagerly(params["eager_run"])

        matrices_list = [[get_response_matrix(self.backend, q) for q in measured_qubits] for measured_qubits in measured_qubits_list]

        ibu_mitigated = []
        for i in range(len(matrices_list)):
            ibu = IBU(matrices_list[i], params)
            ibu.set_obs(dict(counts[i]))
            ibu.initialize_guess()
            # 4x4 first timestep theoretically perfect lattice data, not used
            # t_true_dict = {"0001" : 1/8, "0010" : 1/8, "0101" : 1/8, "0110" : 1/8, "1001" : 1/8, "1010" : 1/8, "1101" : 1/8, "1110" : 1/8}
            t_sol, max_iters, tracker = ibu.train(params["max_iters"], tol=params["tol"])
            
            ibu_mitigated.append({label: int(prob[0] * shots) for label, prob in ibu.guess_as_dict().items()})

        return ibu_mitigated
    
    def mitigate(
        self,
        qcs: list[QuantumCircuit],
        shots: int,
        counts: list,
        ) -> tuple[list, str]:

        # Data shows that performing REM and then IBU yields better results.
        if (self.readout_error_mitigation == False and self.iterative_bayesian_unfolding == False):
            label = f"raw-collisionless-{self.dims[0]}x{self.dims[1]}-ibm-qpu"
            return counts, label
        if (self.readout_error_mitigation == True):
            label = f"rem-collisionless-{self.dims[0]}x{self.dims[1]}-ibm-qpu"
            counts = self.rem(shots, counts)
        if (self.iterative_bayesian_unfolding == True):
            label = f"ibu-collisionless-{self.dims[0]}x{self.dims[1]}-ibm-qpu"
            counts = self.ibu(qcs, shots, counts)
        if (self.readout_error_mitigation == True and self.iterative_bayesian_unfolding == True):
            label = f"rem-ibu-collisionless-{self.dims[0]}x{self.dims[1]}-ibm-qpu"
        return counts, label
        