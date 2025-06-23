# qlbm-mcgill

Welcome to the QLBM McGill-Calcul Quebec Summer Research Project! Here's how to use the resources currently available:

To run a classical collisionless simulation of an 8x8 lattice with D_2Q_8 discretization, running 10 steps with 100 shots per time-step:

```python
>>> from simulation import *
>>> sim = Simulation2D([8,8], 4)
>>> sim.sim_QTM(10, 100)
```
The PyVista animation will be saved to your CWD with the format: ```collisionless-8x8.gif```.

To run a IBM QPU job with the same lattice and discretization, but with 2 steps and 8192 shots per time-step:
(You will need an IBM Quantum Platform ```token``` which is associated with your IBM ID/IBM account.)

```python
>>> fom ibm_qpu import *
>>> runner = IBM_QPU_Runner([8x8], token)
>>> runner.run(2) # shots are automatically set to 8192
# You will now need to check the IBM Quantum Platfrom website and wait until the job is finished.
>>> visualize(2)
```
The PyVista animation will be saved to your CWD with the format: ```collisionless-8x8-ibm-qpu.gif```.
