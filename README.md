# qlbm-mcgill

Welcome to the QLBM McGill-Calcul Quebec Summer Research Project!

All programs made here use the QLBM software framework by [Calin A. Georgescu et. al.](https://arxiv.org/pdf/2411.19439)

IBU implementation uses already existing code by [sidsrinivasan](https://github.com/sidsrinivasan/PyIBU).

This repository has many dependencies, which can be installed by:

```shell
pip install -r requirements.txt
```

QLBM may install an unsupported Qiskit version. If so, this should fix it:

```shell
pip install --force-reinstall qiskit<2.0
```

To run a classical collisionless simulation of an 4x4 lattice with D_2Q_8 discretization, running 10 steps with 1024 shots per time-step:

```python
# must have 'simulation.py' and 'base.py' in current working directory
from simulation import Simulation2D

steps = 10
shots = 1024

sim = Simulation2D([4,4])
sim.make(steps, shots=1024)
```
The PyVista animation will be saved to the CWD with the format: ```collisionless-sim-4x4.gif```.

To run a IBM QPU job with the same lattice and discretization, but with 2 steps and 1024 shots per time-step:
(This requires an IBM ```token``` and ```instance```, which are created on the [IBM Quantum Platform](https://quantum.cloud.ibm.com/)).

```python
# must have 'ibm_qpu.py' and 'base.py' in current working directory
from ibm_qpu import IBM_QPU_Runner
from qiskit_ibm_runtime import QiskitRuntimeService

name = "[custom name, can be set to anything]"
steps = 2
shots = 1024

QiskitRuntimeService.save_account(
  token="[your API token]",
  channel="ibm_cloud", 
  instance="[your instance name]", 
  name=name, 
  overwrite=True,
  set_as_default=True
) # This only needs to be run once, and can thus be omitted on all subsequent runs

runner = IBM_QPU_Runner([4,4], name,
  # Enable error mitigation:
  zero_noise_extrapolation=True,
  equalization=True
)
runner.make(steps, shots=shots)
```

The PyVista animation will be saved to the CWD with the format: ```zne-collisionless-4x4-ibm-qpu_1024_shots.gif```.

Noise can be introduced into a classical simulation, with selectable single and double qubit gate error probabilities:

```python
# must have 'noise_sim.py' and 'base.py' in current working directory
from noise_sim import Noise_Simulation2D

single_prob = 0.002 # Single qubit gate error probability
double_prob = 0.01 # Double qubit gate error probability
steps = 3 # Use a small number of steps, since there is no reinitialization, unlike Simulation2D
shots = 1024

noise_sim = Noise_Simulation2D(single_prob, double_prob, [4,4])
noise_sim.make(steps, shots=shots)
```
The PyVista animation will be saved to the CWD with the format: ```noisy-collisionless-simulation-4x4_0.002-single-0.01-double_1024_shots.gif```.

```python
#
# TODO
#
# ☑: currently implemented in qlbm-mcgill
#
# Implement error mitigation
#   - Calcul Quebec currently implements the following:
#         ☑ Readout Measurement Mitigation: correction of measurement errors.
#           https://qiskit-community.github.io/qiskit-experiments/manuals/measurement/readout_mitigation.html
#         ☑ Iterative Bayesian Unfolding (IBU): iterative technique to find a more precise
#           distribution of results.
#     and are currently developing:
#         ☑ Zero Noise Extrapolation (ZNE): the circuit is ran at different noise levels to
#           extrapolate an ideal result at the zero-noise limit.
#         ☑ Digital Dynamical Decoupling (DDD): a sequence of identity gates is applied to
#           inactive qubits during circuit execution to limit decoherence effects.
#   - Qiskit's Sampler primitive implements:
#         ☑ Dynamical Decoupling
#         ☑ Pauli Gate Twirling
#   - Other mitigation techniques exist, such as
#         - Multiple circuits can be run simultaneously depending on available qubits, which
#           can allow redundancy and/or save on resources
#         - Quantum Repitition Codes, or even Quantum Surface Codes, are also possible to
#           implement: https://github.com/quantumjim/qec_lectures?tab=readme-ov-file
#
# ☑ Introduce a decoherence/noise/error model for classical simulation
#   - This is important for studying how close we are to implementation in NISQ
#
# MonarQRunner: Run jobs on 24 qubit MonarQ, max lattice size 2x2 or maybe 4x2 if lucky
#   - POSTPONED: MonarQ under mainteance, Yukon only 6 qubits so QLBM unfeasable
#   - Need a way to convert Qiskit StepCircuit.circuit array to PennyLane (done in one function)
#   - Post-processing will require conversion of MonarQ PennyLane-formatted result counts to Qiskit
#     for use in QLBM infrastructure.
#
# Implement reinitialization
#   - Use the counts format of list[dict] to get quasi-probs and then convert to statevector
#
# Create a method of easily choosing initial conditions
#   - Gates are applied to grid and velocity qubits to produce initial conditions 
#
# Simulation2D: Include obstacles (will not be run through QPU, purely visual)
#   - Low priority due to obstacles causing too deep of a circuit (200+)
#
# Simulation3D: May be possible with small lattices ond low dicretization (D_3Q_6)
#   - Low priority due to poor visualization. Better software may help.
#
```



