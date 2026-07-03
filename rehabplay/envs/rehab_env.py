"""RehabPlay rehabilitation environment.

This module defines the core Isaac Lab manager-based reinforcement learning
environment for RehabPlay: a simulated robot-assisted physical therapy task
in which a Franka Panda arm guides a patient limb through a prescribed
range-of-motion (ROM) exercise under safety-constrained force control.

Environment design follows Isaac Lab's manager-based workflow
(``isaaclab.envs.ManagerBasedRLEnv``), composed of the following managers:

    * ActionManager      -- robot joint torque commands.
    * ObservationManager -- robot joint state, patient limb state,
                             force-sensor readings, and target ROM goal.
    * RewardManager      -- ROM-progress shaping, force-safety penalties,
                             compensatory-movement penalties, and a
                             target-achievement bonus.
    * TerminationManager -- safety hard-stop (excessive force) and
                             success termination (target ROM reached).

Status
------
Scaffold only. Full implementation is tracked under Week 2 of the project
build plan (see ``docs/build-log.md``). Populating each manager
configuration below is the primary Week 2 deliverable.

References
----------
Isaac Lab manager-based environment tutorials:
    https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class RehabSceneCfg(InteractiveSceneCfg):
    """Scene configuration: robot, patient limb, and sensors.

    Currently populated:
        - Ground plane and dome light (required for any Isaac Sim scene).
        - Franka Panda robot articulation, using Isaac Lab's built-in
          pretrained configuration.

    TODO (Week 2, remaining):
        - Add simplified patient limb (rigid body; soft-body/Newton later).
        - Add force sensor(s) on the robot end-effector / limb contact.
    """

    # Ground plane: a flat static collision surface. Every Isaac Sim scene
    # needs one, or the robot (and anything else with gravity enabled)
    # falls indefinitely.
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Dome light: uniform ambient lighting for the whole scene. Without a
    # light source, the camera/viewer sees a black scene (this doesn't
    # affect physics, only visualization/streaming).
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

    # Robot: Franka Panda, using Isaac Lab's pretrained/pre-configured
    # articulation (FRANKA_PANDA_CFG). This config already defines the
    # robot's default joint positions, actuator stiffness/damping, and
    # USD asset path -- we only need to set where it spawns.
    #
    # "{ENV_REGEX_NS}/Robot" is Isaac Lab's convention for a prim path
    # that resolves correctly whether num_envs is 1 (as it is now, for
    # this initial scene test) or scaled up to thousands of parallel
    # environments later for PPO training.
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


@configclass
class ActionsCfg:
    """Action space configuration.

    TODO (Week 2):
        - Define robot joint torque action term for the Franka Panda arm.
    """


@configclass
class ObservationsCfg:
    """Observation space configuration.

    TODO (Week 2):
        - Robot joint positions.
        - Robot joint velocities.
        - Patient limb joint angle(s).
        - Force sensor reading(s).
        - Target ROM goal.
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations available to the policy network."""

        pass

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward term configuration.

    TODO (Week 2), per project specification:
        - +1 per degree of ROM achieved toward target.
        - -0.5 penalty when applied force exceeds 20 N.
        - -1 penalty for detected compensatory movement.
        - +5 bonus for reaching target ROM.
    """


@configclass
class TerminationsCfg:
    """Termination condition configuration.

    TODO (Week 2):
        - Safety hard stop: episode reset if force exceeds 30 N.
        - Success termination: target ROM achieved.
        - Timeout termination: episode length exceeded.
    """


@configclass
class RehabEnvCfg(ManagerBasedRLEnvCfg):
    """Full environment configuration for RehabPlay.

    Composes the scene, action, observation, reward, and termination
    configurations above into a single Isaac Lab environment config.
    """

    scene: RehabSceneCfg = RehabSceneCfg(num_envs=1, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post-initialization: simulation timestep and episode settings.

        TODO (Week 2):
            - Set ``self.decimation`` and ``self.sim.dt`` appropriately.
            - Set ``self.episode_length_s``.
        """
        pass


class RehabEnv(ManagerBasedRLEnv):
    """RehabPlay rehabilitation task environment.

    Thin wrapper around :class:`isaaclab.envs.ManagerBasedRLEnv`. Custom
    logic beyond manager-based configuration (if any is required) will be
    added here during Week 2 implementation.
    """

    cfg: RehabEnvCfg
