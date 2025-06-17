from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit

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

from os import listdir, chdir

import threading, time

import imageio
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw
from pyvista import themes

class Simulation2D:
    """
    2D simulation with collision (STM) and collisionless (QTM) conditions and no obstacles.
    sim_QTM: collisionless simulation.
    sim_STM: simulation with collision.
    """

    def __init__(self, dims, vs):
        """
        dims: size 2 array denoting x & y
        vs: velocity of both dimensions
        """
        print(f"Preparing {dims[0]}x{dims[1]} simulation...")
        self.dims = dims
        self.qtm_lattice = CollisionlessLattice(
            {
                "lattice": {"dim": {"x": dims[0], "y": dims[1]}, "velocities": {"x": vs, "y": vs}}
            },
        )
        self.stm_lattice = SpaceTimeLattice(
            num_timesteps=1,
            lattice_data={
                "lattice": {"dim": {"x": dims[0], "y": dims[1]}, "velocities": {"x": 2, "y": 2}},
                "geometry": [],
            },
        )

    def timer(self):
        start = int(time.time())
        time.sleep(1)
        while threading.active_count() == 4:
            time.sleep(1)
            diff = int(time.time()) - start
            print("\033[A\033[K\r" + f"Time elapsed: {int(diff/60)} minute(s) and {diff % 60} seconds.")

    def run_sim(self, cfg, lattice, dir, steps, shots):
        cfg.prepare_for_simulation()
        runner = QiskitRunner(
            cfg,
            lattice,
        )
        runner.run(
            steps,  # Number of time steps
            shots,  # Number of shots per time step
            dir,
            statevector_snapshots=True,
        )

    def create_animation(self, simdir, output_filename):
        pv.set_plot_theme(themes.ParaViewTheme())
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

    def sim_QTM(self, steps, shots):
        dir = f"qlbm-output/collisionless-{self.dims[0]}x{self.dims[1]}-qiskit"
        create_directory_and_parents(dir)
        cfg = SimulationConfig(
            initial_conditions=CollisionlessInitialConditions(self.qtm_lattice),
            algorithm=CQLBM(self.qtm_lattice),
            postprocessing=EmptyPrimitive(self.qtm_lattice),
            measurement=GridMeasurement(self.qtm_lattice),
            target_platform="QISKIT",
            compiler_platform="QISKIT",
            optimization_level=0,
            statevector_sampling=True,
            execution_backend=AerSimulator(method="statevector"),
            sampling_backend=AerSimulator(method="statevector"),
        )
        # Simulate the circuits using both snapshots
        print(f"Running {self.dims[0]}x{self.dims[1]} collisionless simulation...\n")
        sim_thread = threading.Thread(target=self.run_sim, args=((cfg, self.qtm_lattice, dir, steps, shots,)))
        time_thread = threading.Thread(target=self.timer)
        sim_thread.start()
        time_thread.start()
        sim_thread.join()
        time_thread.join()
        print("Simulation complete. Creating animation...")
        
        self.create_animation(f"{dir}/paraview", f"collisionless-{self.dims[0]}x{self.dims[1]}.gif")
        print("Animation created and saved.")

    def sim_STM(self, steps, shots):
        dir = f"qlbm-output/collision-{self.dims[0]}x{self.dims[1]}-qiskit"
        create_directory_and_parents(dir)
        cfg = SimulationConfig(
            initial_conditions=PointWiseSpaceTimeInitialConditions(
                self.stm_lattice, grid_data=[((1, 5), (True, True, True, True))]
            ),
            algorithm=SpaceTimeQLBM(self.lattice),
            postprocessing=EmptyPrimitive(self.lattice),
            measurement=SpaceTimeGridVelocityMeasurement(self.lattice),
            target_platform="QISKIT",
            compiler_platform="QISKIT",
            optimization_level=0,
            statevector_sampling=False,
            execution_backend=AerSimulator(method="statevector"),
            sampling_backend=AerSimulator(method="statevector"),
        )
        cfg.prepare_for_simulation()

        runner = QiskitRunner(
            cfg,
            self.stm_lattice,
        )
        print(f"Running {self.dims[0]}x{self.dims[1]} collision simulation...")
        runner.run(
            steps,  # Number of time steps
            shots,  # Number of shots per time step
            dir,
            statevector_snapshots=True,
        )
        print("Simulation complete.")
        pv.set_plot_theme(themes.ParaViewTheme())
        self.create_animation(f"{dir}/paraview", f"collision-{self.dims[0]}-{self.dims[1]}.gif")

sim = Simulation2D([16,8], 4) # Create a new Simulation2D class with specified dimensions and velocities.
sim.sim_QTM(50, 1000) # Run QTM (collisionless) simulation with specified timesteps and snapshots.
# Alternatively:
#sim.sim_STM(50, 1000) # Run STM (with collision) simulation ' ' ' ' '