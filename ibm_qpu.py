from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, ClassicalRegister

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import Sampler

from qlbm.components import (
    CQLBM,
    CollisionlessInitialConditions,
    EmptyPrimitive,
    GridMeasurement,
)
from qlbm.components import EmptyPrimitive
from qlbm.components.spacetime import (
    PointWiseSpaceTimeInitialConditions,
    SpaceTimeGridVelocityMeasurement,
    SpaceTimeQLBM,
)
from qlbm.lattice import SpaceTimeLattice
from qlbm.infra import QiskitRunner, SimulationConfig
from qlbm.lattice import CollisionlessLattice
from qlbm.tools.utils import create_directory_and_parents
from qlbm.infra.result import CollisionlessResult

from os import listdir, chdir

import threading, time

import imageio
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw
from pyvista import themes

from qlbm.infra.reinitialize import CollisionlessReinitializer
from qiskit.quantum_info import Statevector

def create_animation(simdir, output_filename):
    vti_files = sorted(
        [f"{simdir}/{fname}" for fname in listdir(simdir) if fname.endswith(".vti")]
    )
    stl_mesh = pv.read(
        [f"{simdir}/{fname}" for fname in listdir(simdir) if fname.endswith(".stl")]
    )

    # Find the global maximum scalar value
    max_scalar = 0
    for vti_file in vti_files:
        mesh = pv.read(vti_file)
        if mesh.active_scalars is not None:
            max_scalar = max(max_scalar, mesh.active_scalars.max())

    images = []
    sargs = dict(
        title="Measurements at gridpoint",
        title_font_size=20,
        label_font_size=16,
        shadow=True,
        n_labels=3,
        italic=True,
        fmt="%.1f",
        font_family="arial",
        position_x=0.2,  # Centering the scalar bar
        position_y=0.05,
    )

    images = []
    for c, vti_file in enumerate(vti_files):
        time_step_mesh = pv.read(vti_file)

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(
            time_step_mesh,
            clim=[0, max_scalar],
            show_edges=True,
            scalar_bar_args=sargs,
        )

        plotter.add_mesh(
            stl_mesh,
            show_scalar_bar=False,
        )
        plotter.view_xy()
        img = plotter.screenshot(
            transparent_background=True,
        )
        images.append(img)

        # Clean up the plotter
        plotter.close()

        # Convert screenshot to PIL image
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        # Draw progress bar
        bar_width = int(pil_img.width * 0.8)
        bar_height = 20
        bar_x = (pil_img.width - bar_width) // 2
        bar_y = pil_img.height - 40
        progress = int((c + 1) / (len(vti_files)) * bar_width)

        draw.rectangle(
            [bar_x, bar_y, bar_x + bar_width, bar_y + bar_height],
            outline="black",
            width=3,
        )
        draw.rectangle(
            [bar_x, bar_y, bar_x + progress, bar_y + bar_height], fill="purple"
        )

        images.append(np.array(pil_img))

    # Create the GIF from the collected images
    imageio.mimsave(output_filename, images, duration=1, loop=0)

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

    def __init__(self, dims, token):
        
        print(f"Initializing {dims[0]}x{dims[1]} runner... ", end="")
        self.service = QiskitRuntimeService(channel="ibm_quantum", 
                               token=token)
        
        self.dims = dims
        self.lattice = CollisionlessLattice(
            {
            "lattice": {"dim": {"x": dims[0], "y": dims[1]}, "velocities": {"x": 4, "y": 4}},
            }
        )
        print("done.")

    def run(self, steps, shots=8192):

        step_qcs = [StepCircuit(self.lattice, i).circuit for i in range(steps+1)]

        backend = QiskitRuntimeService().least_busy(simulator=False, operational=True, min_num_qubits=25)
        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)

        print(f"Sending {steps} step job to {backend} with {shots} shots per time-step... ", end="")

        qcs = [pass_manager.run(qc) for qc in step_qcs]

        sampler = Sampler(backend)

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

