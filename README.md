# qlbm-mcgill

Welcome to the QLBM McGill-Calcul Quebec Summer Research Project!

All programs made here use the QLBM software framework by [Calin A. Georgescu et. al.](https://arxiv.org/pdf/2411.19439)

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
(This requires an IBM ```token``` and ```instance```, which are created on the [IBM Quantum Platform](https://quantum.cloud.ibm.com/).)

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

runner = IBM_QPU_Runner([4,4], name)
runner.make(steps, shots=shots)
# To enable Readout Error Mitigation:
# runner.make(steps, shots=shots, readout_error_mitigation=True)
# This will cause the filename to change to "mitigated-collisionless-4x4-ibm-qpu_1024_shots.gif"
```

Some jobs may take a while due to long queues, so the job id may be used instead to create the animation, provided the job is complete:

```python
# assuming a KeyboardInterrupt of the previous block of code, so "name" is associated to an IBM account
job_id = "[job id]"
runner = IBM_QPU_Runner([4,4], name)
runner.job_id = job_id
runner.visualize(steps, shots=shots)
```
The PyVista animation will be saved to the CWD with the format: ```collisionless-4x4-ibm-qpu_1024_shots.gif```.

Noise can be introduced into a classical simulation, with selectable single and double qubit gate error probabilities:

```python
# must have 'noise_sim.py' and 'base.py' in current working directory
from noise_sim import Noise_Simulation2D

single_prob = 0.002 # Single qubit gate error probability
double_prob = 0.01 # Double qubit gate error probability
steps = 3 # Use a small number of shots, since there is no reinitialization, unlike Simulation2D
shots = 1024

noise_sim = Noise_Simulation2D(single_prob, double_prob, [4,4])
noise_sim.make(steps, shots=shots)
```
The PyVista animation will be saved to the CWD with the format: ```noisy-collisionless-simulation-4x4_0.002-single-0.01-double.gif```.

```python
#
# TODO
#
# Implement error mitigation
#   - Calcul Quebec currently implements the following:
#         ☑ Readout Measurement Mitigation: correction of measurement errors.
#           https://qiskit-community.github.io/qiskit-experiments/manuals/measurement/readout_mitigation.html
#         - Iterative Bayesian Unfolding (IBU): iterative technique to find a more precise
#           distribution of results.
#     and are currently developing:
#         - Zero Noise Extrapolation (ZNE): the circuit is ran at different noise levels to
#           extrapolate an ideal result at the zero-noise limit.
#         - Digital Dynamical Decoupling (DDD): a sequence of identity gates is applied to
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
#   - Should be possible using Qiskit's SamplerOptions
#
# MonarQRunner: Run jobs on 24 qubit MonarQ, max lattice size 2x2 or maybe 4x2 if lucky
#   - POSTPONED: MonarQ under mainteance, Yukon only 6 qubits so QLBM unfeasable
#   - Need a way to convert Qiskit StepCircuit.circuit array to PennyLane (done in one function)
#   - Post-processing will require conversion of MonarQ PennyLane-formatted result counts to Qiskit
#     for use in QLBM infrastructure.
#
# Use Quantum State Tomography (QST) to reinitialize circuits to avoid depth limits
#     https://qiskit-community.github.io/qiskit-experiments/manuals/verification/state_tomography.html
#   - QST requires 4^n measurements for n qubits, so it may be impossible at larger scale
#         - 4x2 (smallest visualizable lattice): 5 qubits to measure (3 grid, 2 velocity)
#   - We will first require decent results from 1 time-step run on a QPU,
#     from there use QST to get the estimated statevector of the grid and velocity qubits,
#     onto which |0>s are prepended and appended for ancillae, and then used for reinitialization
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
