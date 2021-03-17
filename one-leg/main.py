from simulation import Simulation

sim = None


def initialize_env():
    global sim
    sim = Simulation()


def simulate():
    sim.simulate()


if __name__ == "__main__":
    # execute only if run as a script
    initialize_env()
    simulate()
