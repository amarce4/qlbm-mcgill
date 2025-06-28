from base import *

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
        """
        Basic timer, to be run in a separate thread, to track how long the simulation takes.
        """
        start = int(time.time())
        time.sleep(1)
        while threading.active_count() == 4:
            time.sleep(1)
            diff = int(time.time()) - start
            print("\033[A\033[K\r" + f"Time elapsed: {int(diff/60)} minute(s) and {diff % 60} seconds.")

    def run_sim(self, cfg, lattice, dir, steps, shots):
        """
        Helper function; runs a Qiskit simulation given config, lattice, directory, number of steps and shots.
        """
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

    def sim_QTM(self, steps, shots):#, initial_conditions=None):
        """
        Performs a Quantum Transport Method (collisionless) simulation, given number of steps and shots per time-step.
        """
        dir = f"qlbm-output/collisionless-{self.dims[0]}x{self.dims[1]}-qiskit"
        create_directory_and_parents(dir)
        # if initial_conditions == None:
        #         initial_conditions=CollisionlessInitialConditions(self.qtm_lattice),
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
            # sampling_backend=None
            sampling_backend=AerSimulator(method="statevector"),
        )
        # Simulate the circuits using both snapshots
        print(f"Running {self.dims[0]}x{self.dims[1]} collisionless simulation...\n")
        # sim_thread = threading.Thread(target=self.run_sim, args=((cfg, self.qtm_lattice, dir, steps, shots,)))
        # time_thread = threading.Thread(target=self.timer)
        # sim_thread.start()
        # time_thread.start()
        # sim_thread.join()
        # time_thread.join()
        self.run_sim(cfg, self.qtm_lattice, dir, steps, shots)
        print("Simulation complete. Creating animation...")
        create_animation(f"{dir}/paraview", f"collisionless-{self.dims[0]}x{self.dims[1]}.gif")
        print("Animation created and saved.")

    def sim_STM(self, steps, shots):
        """
        Performs a Space-Time Method (with collision) simulation.
        !!! Only compatible with D_2Q_4 (vs = 2) discretization !!!
        """
        dir = f"qlbm-output/collision-{self.dims[0]}x{self.dims[1]}-qiskit"
        create_directory_and_parents(dir)
        cfg = SimulationConfig(
            initial_conditions=PointWiseSpaceTimeInitialConditions(
                self.stm_lattice, grid_data=[((1, 5), (True, True, True, True))]
            ),
            algorithm=SpaceTimeQLBM(self.stm_lattice),
            postprocessing=EmptyPrimitive(self.stm_lattice),
            measurement=SpaceTimeGridVelocityMeasurement(self.stm_lattice),
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
        create_animation(f"{dir}/paraview", f"with-collision-{self.dims[0]}-{self.dims[1]}.gif")