# qlbm-mcgill

Welcome to the QLBM McGill-Calcul Quebec Summer Research Project!

All programs made here use the QLBM software framework by [Calin A. Georgescu et. al.](https://arxiv.org/pdf/2411.19439)

To run a classical collisionless simulation of an 8x8 lattice with D_2Q_8 discretization, running 10 steps with 100 shots per time-step:

```python
# must have 'simulation.py' and 'base.py' in current working directory
from simulation import Simulation2D
sim = Simulation2D([8,8], 4)
sim.sim_QTM(10, 100)
# sim.sim_STM() can do simulations with collision, but only at a D_2Q_4 discretization
```
The PyVista animation will be saved to your CWD with the format: ```collisionless-8x8.gif```.

To run a IBM QPU job with the same lattice and discretization, but with 2 steps and 8192 shots per time-step:
(You will need an IBM Quantum Platform ```token``` which is associated with your IBM ID/IBM account.)

```python
# must have 'ibm_qpu.py' and 'base.py' in current working directory
from ibm_qpu import IBM_QPU_Runner
token = "[your API token]"
runner = IBM_QPU_Runner([8,8], token)
runner.make(2) # runner.make(2, shots=8192)
```

Some jobs may take a while due to long queues, so you can use the job id to create the animation, provided the job is complete:

```python
job_id = "[job id]"
token = "[your API token]"
runner = IBM_QPU_Runner([8,8], token)
runner.job_id = job_id
runner.visualize(2)
```
The PyVista animation will be saved to your CWD with the format: ```collisionless-8x8-ibm-qpu.gif```.

```python
#
# TODO
#
# All: Implement error mitigation through post-processing techniques
#   - Refer to Calcul Quebec coding demo
#
# All: Create a method of easily choosing initial conditions
#   - Gates are applied to grid and velocity qubits to produce initial conditions 
#
# MonarQRunner: Run jobs on 24 qubit MonarQ, max lattice size 256x128 or 8x8x8 collisionless
#   - POSTPONED: MonarQ under mainteance, Yukon only 6 qubits so QLBM unfeasable
#   - Need a way to convert Qiskit StepCircuit.circuit array to PennyLane (done in one function)
#   - Post-processing will require conversion of MonarQ PennyLane-formatted result counts to Qiskit
#     for use in QLBM infrastructure.
#
# All: Use Quantum State Topography (QST) to reinitialize circuits to avoid depth limits
#   - QST requires 4^n measurements for n qubits, so it may be impossible at this scale
#
# Simulation2D: Include obstacles (will not be run through QPU, purely visual)
#   - Low priority due to obstacles causing too deep of a circuit (200+)
#
# Simulation3D: May be possible with small lattices ond low dicretization (D_3Q_6)
#   - Low priority due to poor visualization. Better software may help.
#
```
