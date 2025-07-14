from base import *

class Simulation2D(Runner):
    """
    2D simulation with collision (STM) and collisionless (QTM) conditions and no obstacles.
    
    Attributes:
        lattice: Lattice | SpaceTimeLattice
        dims: tuple | list, x & y dimensions
        collision: bool, whether the lattice has collision
        label: str, name of the simulation file
        active: bool, whether or not the simulation is actively running (for the timer)
    """

    lattice: Lattice | SpaceTimeLattice
    dims: tuple | list
    collision: bool
    label: str
    active: bool

    def __init__(
        self, 
        dims: tuple | list, 
        vs: tuple | list = [4,4], 
        collision: bool = False
        ) -> None:
        """
        vs: tuple | list, x & y velocities, default to [4,4]
        """
        print(f"Preparing {dims[0]}x{dims[1]} simulation...")
        self.active = False
        self.dims = dims
        self.collision = collision
        if collision:
            self.lattice = SpaceTimeLattice(
            num_timesteps=1,
            lattice_data={
                "lattice": {"dim": {"x": dims[0], "y": dims[1]}, "velocities": {"x": 2, "y": 2}},
                "geometry": [],
                },
            )
        else:
            self.lattice = Lattice(dims, vs)

    @override
    def run(
        self, 
        steps: int, 
        shots: int = DEFAULT_SHOTS
        ) -> None:
        if self.collision:
            print(f"Running {self.dims[0]}x{self.dims[1]} simulation with collision...\n")
            self.label = f"collision-sim-{self.dims[0]}x{self.dims[1]}"
        else:
            print(f"Running {self.dims[0]}x{self.dims[1]} collisionless simulation...\n")
            self.label = f"collisionless-sim-{self.dims[0]}x{self.dims[1]}"
        
        dir = f"qlbm-output/{self.label}"
        rmdir_rf(dir)
        create_directory_and_parents(dir)
        
        if self.collision:
            cfg = SimulationConfig(
                initial_conditions=PointWiseSpaceTimeInitialConditions(
                    self.lattice, grid_data=[((1, 5), (True, True, True, True))]
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
        else:
            cfg = SimulationConfig(
                initial_conditions=CollisionlessInitialConditions(self.lattice),
                algorithm=CQLBM(self.lattice),
                postprocessing=EmptyPrimitive(self.lattice),
                measurement=GridMeasurement(self.lattice),
                target_platform="QISKIT",
                compiler_platform="QISKIT",
                optimization_level=0,
                statevector_sampling=True,
                execution_backend=AerSimulator(method="statevector"),
                # sampling_backend=None
                sampling_backend=AerSimulator(method="statevector"),
            )
        
        sim_thread = threading.Thread(target=self.sim, args=((cfg, steps, shots,)))
        time_thread = threading.Thread(target=self.timer)
        sim_thread.start()
        time_thread.start()
        sim_thread.join()
        time_thread.join()
        print("Simulation complete.")

    @override
    def visualize(self) -> str:
        print("Visualizing... ", end="")
        pv.set_plot_theme(themes.ParaViewTheme())
        create_animation(f"qlbm-output/{self.label}/paraview", f"{self.label}.gif")
        print("done.")
        return f"{self.label}.gif"

    @override
    def make(self, 
        steps: int, 
        shots: int = DEFAULT_SHOTS
        ) -> str:
        self.run(steps, shots=shots)
        vis = self.visualize()
        return vis

    def timer(self):
        """
        Basic timer, to be run in a separate thread, to track how long the simulation takes.
        """
        start = int(time.time())
        time.sleep(1)
        while self.active:
            time.sleep(0.9)
            diff = int(time.time()) - start
            print("\r", end="")
            print(f"Time elapsed: {int(diff/60)} minute(s) and {diff % 60} second(s).", end="")
        print()

    def sim(self, 
            cfg: SimulationConfig, 
            steps: int, 
            shots: int = DEFAULT_SHOTS
        ) -> None:
        self.active = True
        """
        Helper function; runs a Qiskit simulation given config, lattice, directory, number of steps and shots.
        """
        cfg.prepare_for_simulation()
        runner = QiskitRunner(
            cfg,
            self.lattice,
        )
        runner.run(
            steps,  # Number of time steps
            shots,  # Number of shots per time step
            f"qlbm-output/{self.label}",
            statevector_snapshots=True,
        )
        self.active = False