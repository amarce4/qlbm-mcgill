from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, ClassicalRegister

from qiskit_ibm_runtime import QiskitRuntimeService, IBMBackend
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer.primitives import SamplerV2 as SimSampler
from qiskit_aer.primitives import EstimatorV2 as SimEstimator
from qiskit_ibm_runtime import SamplerOptions
# from qiskit_aer.primitives import EstimatorV2 as Estimator
# from qiskit_ibm_runtime import EstimatorOptions

from qiskit import transpile
from qiskit_aer.noise import NoiseModel, depolarizing_error

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
from qlbm.infra.result import CollisionlessResult, SpaceTimeResult

from os import listdir, chdir, path
from shutil import rmtree

import threading, time

import imageio
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw
from pyvista import themes

from qlbm.infra.reinitialize import CollisionlessReinitializer

from IPython.display import Image as Draw

from abc import ABC, abstractmethod
from typing_extensions import override

class Runner(ABC):
    
    lattice: CollisionlessLattice
    dims: tuple | list

    def __init__(self):
        super().__init__()

    @abstractmethod
    def run(
        self, 
        steps: int, 
        shots: int = 1024):
        """
        Runs "steps" number of CQLBM algorithm circuits with "shots" shots.
        """
        pass

    @abstractmethod
    def visualize(
        self,
        steps: int,
        shots: int = 1024):
        """
        Visualizes the data in a ".gif" file.
        """
        pass

    @abstractmethod
    def make(
        self,
        steps: int,
        shots: int = 1024):
        """
        Runs and visualizes the lattice.
        """
        pass


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
                 collision: bool = False
            ) -> None:
        if collision == False:
            if type(init_cond == None):
                self.circuit = CollisionlessInitialConditions(lattice).circuit
            else:
                self.circuit = init_cond
            for i in range(0, num_steps):
                self.circuit.compose(CQLBM(lattice).circuit, inplace=True)
                self.circuit.reset([-3,-4,0,1]) # ancilla qubits
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
     
class Lattice(CollisionlessLattice):

    dims: list | tuple
    
    def __init__(
            self, 
            dims: list | tuple, 
            vs: list | tuple = [4,4]
        ) -> None:
        super().__init__(
            {
            "lattice": {"dim": {"x": dims[0], "y": dims[1]}, "velocities": {"x": vs[0], "y": vs[1]}},
            }
        )

def rmdir_rf(dir: str):
    try:
        rmtree(dir)
    except OSError:
        pass

def create_animation(simdir: str, output_filename: str):
    """
    Creates a PyVista animation given the directory in which simulation '.vti' files are stored.
    (NOT MY WORK: credit to QLBM)
    """
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
    imageio.mimsave(output_filename, images, duration=6, loop=0)

def count_gates(qc: QuantumCircuit):
    """
    Helper function for remove_idle_wires()
    (NOT MY WORK: credit to Qiskit)
    """
    gate_count = { qubit: 0 for qubit in qc.qubits }
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit] += 1
    return gate_count

def remove_idle_wires(qc: QuantumCircuit):
    """
    Removes any wires that do not have any gates acting upon them.
    This is useful for ibm_qpu.StepCircuit, since extra ancilla qubits are added to the circuit 
    for use in obstacle operators. Since we are not using obstacles, these qubits can be removed.
    (NOT MY WORK: credit to Qiskit)
    """
    qc_out = qc.copy()
    gate_count = count_gates(qc_out)
    for qubit, count in gate_count.items():
        if count == 0:
            qc_out.qubits.remove(qubit)
    return qc_out