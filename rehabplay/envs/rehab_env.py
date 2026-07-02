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


@configclass
class RehabSceneCfg(InteractiveSceneCfg):
    """Scene configuration: robot, patient limb, and sensors.

    TODO (Week 2):
        - Add Franka Panda articulation from Isaac Lab's built-in assets.
        - Add simplified patient limb (rigid body; soft-body/Newton later).
        - Add force sensor(s) on the robot end-effector / limb contact.
        - Add ground plane and lighting.
    """


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
