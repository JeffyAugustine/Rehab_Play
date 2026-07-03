"""Standalone smoke test: verify RehabSceneCfg loads without errors.

This script does NOT run the full RehabEnv (that requires the action /
observation / reward / termination managers to be implemented first,
which is later in the Week 2 plan). It only verifies that the scene --
ground plane, lighting, the patient torso, the Franka Panda robot, and
the articulated patient arm -- spawns correctly in Isaac Sim, and that
the patient arm's joints report sensible names, initial positions, and
clinical ROM limits.

Usage
-----
    python scripts/test_scene.py --headless

Expected output
----------------
    [INFO]: Scene loaded successfully.
    [INFO]: Robot joint names: [...]
    [INFO]: Robot default joint positions: [...]
    [INFO]: Patient arm joint names: ['shoulder_flexion_joint', 'shoulder_abduction_joint', 'shoulder_rotation_joint', 'elbow_flexion_joint', 'forearm_rotation_joint', 'wrist_flexion_joint', 'wrist_deviation_joint']
    [INFO]: Patient arm initial joint positions: [...]
    [INFO]: Patient arm joint limits (lower, upper): [...]

followed by the simulation closing itself after a short run.
"""

import argparse

from isaaclab.app import AppLauncher

# --- CLI arguments -----------------------------------------------------
# AppLauncher.add_app_launcher_args() injects the standard Isaac Sim
# launch flags (--headless, --device, etc.) so this script behaves
# consistently with Isaac Lab's own example scripts.
parser = argparse.ArgumentParser(description="RehabPlay scene smoke test.")
parser.add_argument(
    "--num_steps",
    type=int,
    default=100,
    help="Number of simulation steps to run before exiting.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --- Launch Isaac Sim ---------------------------------------------------
# Isaac Sim must be launched BEFORE importing any isaaclab modules that
# touch the simulator (isaaclab.sim, isaaclab.assets, isaaclab.scene).
# This is a hard Isaac Lab requirement, not a style choice.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports that require the simulation app to already be running -----
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from rehabplay.envs.rehab_env import RehabSceneCfg


def main() -> None:
    """Spawn the RehabPlay scene and step the simulation briefly."""

    # SimulationCfg controls the physics timestep (dt) and rendering.
    # dt=1/120 (120 Hz physics) is Isaac Lab's common default for
    # manipulation tasks -- fine enough for stable contact/force
    # simulation between the robot and (eventually) the patient limb.
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)

    # Point the default camera at roughly the center of the scene (robot
    # + torso + arm cluster), so if you're viewing this through the
    # Isaac Sim streaming viewer, you see everything at once instead of
    # having to manually navigate. Raised target height slightly to
    # account for the taller torso/higher shoulder point.
    sim.set_camera_view(eye=[2.5, 2.0, 2.0], target=[0.5, 0.0, 0.55])

    # Build the scene from our config. num_envs=1 here since this is a
    # single-instance visual/structural smoke test, not vectorized
    # training (that comes later, with num_envs in the thousands).
    scene_cfg = RehabSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    # Reset must be called once before stepping -- this initializes all
    # physics handles for the assets defined in the scene config.
    sim.reset()
    print("[INFO]: Scene loaded successfully.")
    print(f"[INFO]: Robot joint names: {scene['robot'].joint_names}")
    print(
        "[INFO]: Robot default joint positions: "
        f"{scene['robot'].data.default_joint_pos}"
    )
    print(f"[INFO]: Patient arm joint names: {scene['patient_arm'].joint_names}")
    print(
        "[INFO]: Patient arm initial joint positions: "
        f"{scene['patient_arm'].data.joint_pos}"
    )
    print(
        "[INFO]: Patient arm joint limits (lower, upper): "
        f"{scene['patient_arm'].data.joint_pos_limits}"
    )

    # sim.get_physics_dt() reads back the ACTIVE physics timestep from the
    # running simulation context, rather than assuming the value we
    # requested in SimulationCfg was applied exactly as given. This is
    # the pattern Isaac Lab's own tutorials use for this reason.
    sim_dt = sim.get_physics_dt()

    # Step the simulation a fixed number of times with zero action targets,
    # just to confirm the robot holds a stable pose and nothing crashes or
    # explodes (a common symptom of a malformed articulation config).
    for step in range(args_cli.num_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    print(f"[INFO]: Completed {args_cli.num_steps} simulation steps without error.")


if __name__ == "__main__":
    main()
    simulation_app.close()
