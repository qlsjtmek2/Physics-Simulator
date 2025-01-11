import Simulator as sim
import UI as ui


simulator_manager = sim.SimulatorManager()
ui_manager = ui.UIManager(simulator_manager)

# simulator_manager.add_simulator(
#     "Simple Harmonic Motion 3D", sim.SimpleHarmonicSimulator3D())
simulator_manager.add_simulator(
    "Simple Harmonic", sim.SimpleHarmonicSimulator())
simulator_manager.add_simulator(
    "Damped Harmonic", sim.DampedHarmonicSimulator())
simulator_manager.add_simulator(
    "Free Fall", sim.FreeFallSimulator())

simulator_manager.add_simulator(
    "Pendulum", sim.PendulumSimulator())
simulator_manager.add_simulator(
    "Damped Pendulum", sim.DampedPendulumSimulator())
simulator_manager.add_simulator(
    "Circular", sim.CircularMotionSimulator())
simulator_manager.add_simulator(
    "Projectile", sim.ProjectileSimulator())
simulator_manager.add_simulator(
    "Damped Projectile", sim.DampedProjectileSimulator())

ui_manager.start()
