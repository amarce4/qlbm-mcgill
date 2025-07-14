from base import *

rem_table = {
    "4x2": "d1oniu0t0npc73fltdqg",
    "4x4": "d1onjl15jdrc73drondg"
}

class ErrorMitigator:

    dims: list | tuple
    lattice: CollisionlessLattice | Lattice
    readout_error_mitigation: bool

    def __init__(
            self,
            lattice: CollisionlessLattice | Lattice,
            readout_error_mitigation: bool = False
            ) -> None:
        self.lattice = lattice
        self.dims = lattice.dims
        self.readout_error_mitigation = readout_error_mitigation

    def readout_error(
            self,
            shots: int,
            counts: list,
            backend: IBMBackend,
            use_table: bool = True
        ) -> tuple[list, str]:

        if self.readout_error_mitigation:
            
            print("Mitigating readout error...", end="")

            label = f"mitigated-collisionless-{self.dims[0]}x{self.dims[1]}-ibm-qpu"
            measured_qubits = StepCircuit(self.lattice, 1).grid_qubits
            exp = CorrelatedReadoutError(measured_qubits)
            
            
            
            if (use_table == True and self.dims[0] == 4 and self.dims[1] == 2):
                with open("rem-table/4x2.json", "r") as file:
                    raw = json.load(file, cls=RuntimeDecoder)
                data = ExperimentData(experiment=exp)
                data._add_result_data(raw, rem_table["4x2"])
                result = exp.analysis.run(data)
            elif (use_table == True and self.dims[0] == 4 and self.dims[1] == 4):
                with open("rem-table/4x4.json", "r") as file:
                    raw = json.load(file, cls=RuntimeDecoder)
                data = ExperimentData(experiment=exp)
                data._add_result_data(raw, rem_table["4x4"])
                result = exp.analysis.run(data)
            else:
                exp.set_run_options(shots=512)
                result = exp.run(backend)
                
            mitigator = result.analysis_results("Correlated Readout Mitigator", dataframe=True).iloc[0].value
            mitigated_quasi_probs = [mitigator.quasi_probabilities(count) for count in counts]

            mitigated_probs = [(prob.nearest_probability_distribution().binary_probabilities()) for prob in mitigated_quasi_probs]

            mitigated_counts = [{label: int(prob * shots) for label, prob in mitigated_prob.items()} for mitigated_prob in mitigated_probs]
        else:
            label = f"collisionless-{self.dims[0]}x{self.dims[1]}-ibm-qpu"
            mitigated_counts = counts
        return mitigated_counts, label
        

    