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

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
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

# Path to the bundled patient-arm URDF, resolved relative to this file so
# it works regardless of the current working directory the script is run
# from.
_PATIENT_ARM_URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "assets", "patient_arm.urdf"
)


@configclass
class RehabSceneCfg(InteractiveSceneCfg):
    """Scene configuration: robot, patient arm, and sensors.

    Currently populated:
        - Ground plane and dome light (required for any Isaac Sim scene).
        - Patient torso: a static box anchoring the patient arm's
          shoulder and providing real collision geometry for the robot
          to avoid.
        - Franka Panda robot articulation, using Isaac Lab's built-in
          pretrained configuration.
        - Patient arm: an anatomically-complete, 7-joint (shoulder x3,
          elbow x2, wrist x2) articulated arm with real clinical ROM
          limits and passive joint resistance. See
          ``rehabplay/assets/patient_arm.urdf`` for the full anatomical
          design rationale and per-joint citations.

    TODO (Week 2, remaining):
        - Add force sensor(s) on the robot end-effector / limb contact
          (requires enabling contact reporting on both bodies).
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

    # Patient torso: a static, fixed box standing in for a seated
    # patient's body. Two reasons this exists, not just cosmetics:
    #   1. Without it, the patient arm's shoulder is anchored to an
    #      invisible point in mid-air -- physically nonsensical and
    #      visually confusing.
    #   2. It gives the shoulder's joint limits a REAL physical surface
    #      to (eventually) collide with, rather than only a numeric
    #      limit. Real shoulder ROM is partly bounded by the arm
    #      physically contacting the torso (e.g. adduction across the
    #      chest) -- our joint limits already encode the *result* of
    #      that (from clinical goniometry data), but having an actual
    #      solid torso means extreme movements are also stopped by real
    #      collision, and -- importantly for safety -- it gives the
    #      Franka Panda robot something concrete it must avoid colliding
    #      with during therapy movements.
    #
    # Modeled as a single box (not a detailed body shape) since finer
    # torso geometry has no bearing on this project's RL task -- it only
    # needs to (a) look physically plausible and (b) provide correct
    # collision extents.
    #
    # Static, not dynamic: no rigid_props/mass are set, matching the
    # ground plane's pattern above. A patient's torso isn't something
    # the robot should ever be pushing around -- it must stay fixed
    # regardless of any contact force, so it is spawned as static
    # collision geometry rather than a physics-simulated rigid body.
    #
    # Dimensions approximate a seated adult torso:
    #   - depth (x)  = 0.25 m  -- front-to-back chest depth.
    #   - width (y)  = 0.40 m  -- shoulder-to-shoulder width.
    #   - height (z) = 0.65 m  -- hip/seat to shoulder-top, seated.
    #     (Raised from an earlier 0.55 m: with the elbow bent ~90 deg and
    #     the wrist drooping under gravity, the hand's lowest point can
    #     drop roughly 0.47 m below the shoulder -- a shorter torso put
    #     the hand below the ground plane. 0.65 m gives real clearance.)
    #
    # Position: recentered directly in front of the robot (y=0, same axis
    # as the robot base) rather than off to one side -- an earlier
    # placement put the torso/arm off-center, which visually looked like
    # it was on the "wrong side" relative to the robot's reach. Also
    # pushed further out along x (0.75 m from the robot base) to leave
    # clear working space between the robot and the patient, while
    # staying within the Franka Panda's ~0.85 m reach envelope. Still a
    # starting placement -- expect to fine-tune once visually confirmed.
    patient_torso: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PatientTorso",
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.40, 0.65),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.6, 0.65),  # neutral clothing-gray
                metallic=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.75, 0.0, 0.375)),
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
    #
    # NOTE on orientation: the robot is left at its default orientation
    # (no rotation override), which for the Franka Panda's standard
    # asset conventionally reaches outward along +X -- the same
    # direction the patient torso is now placed in. If, once viewed, the
    # robot's reach still doesn't line up with the patient (e.g. it
    # reaches to the side rather than toward the torso), the fix is a
    # one-line addition here: an explicit `rot=` in an InitialStateCfg
    # rotating the robot's yaw by 180 deg (or whatever offset is needed)
    # -- flagging this now rather than guessing blind, since the correct
    # value can only be confirmed by looking at the rendered scene.
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # Patient arm: a clinically-informed, ANATOMICALLY COMPLETE 7-DOF
    # articulated model of a human arm -- matching the real degrees of
    # freedom of the shoulder (3: flexion/extension, abduction/adduction,
    # internal/external rotation), elbow (2: flexion/extension,
    # pronation/supination), and wrist (2: flexion/extension,
    # radial/ulnar deviation).
    #
    # This replaces an earlier free-floating single-capsule version (no
    # anatomical constraints at all) and an intermediate 3-DOF version
    # (only the primary axis of each joint). The full 7-DOF version:
    #   - Supports every real arm movement, not just the three exercise
    #     types named in the initial project brief -- future exercises
    #     (e.g. forearm rotation therapy, shoulder abduction therapy)
    #     need no structural changes to this model.
    #   - Uses real clinical goniometry ROM limits as hard joint limits
    #     on every axis (see patient_arm.urdf for full per-joint values
    #     and citations).
    #   - Applies passive joint damping/friction; the elbow flexion axis
    #     is directly grounded in published relaxed-muscle stiffness
    #     data, other axes are documented, flagged approximations scaled
    #     from that data (see patient_arm.urdf comments for the honest
    #     breakdown of which numbers are sourced vs. approximated).
    #   - Is intentionally NOT a muscle-actuated musculoskeletal model
    #     (see MyoSuite/OpenSim for that level of fidelity) -- judged out
    #     of scope for this project's timeline.
    #
    # The arm is defined in a standalone URDF file (rehabplay/assets/
    # patient_arm.urdf) and loaded here via Isaac Lab's URDF spawner,
    # the same mechanism used to import real robot descriptions. Each
    # multi-axis joint (shoulder, elbow, wrist) is built in the URDF as a
    # stack of single-axis revolute joints connected by small,
    # near-massless structural links -- URDF has no native multi-axis
    # joint type, and this decomposition is the standard approach (also
    # consistent with how biomechanics literature itself models these
    # joints: as orthogonal, intersecting rotation axes).
    patient_arm: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/PatientArm",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=_PATIENT_ARM_URDF_PATH,
            # Anchors the root link (the shoulder attachment point) fixed
            # in space, representing attachment to a stationary (not
            # simulated) torso.
            fix_base=True,
            # No active motor drive on any joint -- the arm is passive.
            # Movement comes only from external contact forces (the
            # robot) and gravity, matching a relaxed patient limb being
            # guided by a therapy robot rather than moving under its own
            # power. Passive resistance instead comes from each joint's
            # <dynamics damping="..." friction="..."/> values, defined
            # directly in the URDF.
            joint_drive=None,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.1,
                angular_damping=0.1,
                max_linear_velocity=5.0,
                max_angular_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        # Initial position: shoulder anchor placed at the torso's
        # front-top-right corner (torso is now at (0.75, 0.0, 0.375),
        # size 0.25 x 0.40 x 0.65 -- front face at x=0.625, top at
        # z=0.70, right edge at y=0.20). Raised from an earlier z=0.425
        # specifically to give the drooping hand clearance above the
        # ground plane (see patient_torso height comment above for the
        # clearance math). A starting placement, not final -- expect to
        # tune once visually confirmed.
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.625, 0.20, 0.70),
            joint_pos={
                # Resting pose: a seated patient's arm resting naturally,
                # NOT a fully straight arm hanging at the side. Per
                # explicit design intent: shoulder relaxed (upper arm
                # roughly vertical), elbow bent to roughly 90 deg
                # (forearm resting forward, e.g. on an armrest/lap), and
                # the wrist/hand drooping down under gravity from that
                # forearm position (unsupported wrist relaxes into
                # flexion). This matches a natural seated resting
                # posture much more closely than a stiff, fully-extended
                # arm would.
                #
                # NOTE: with the corrected "hangs down at zero" geometry
                # (see patient_arm.urdf), shoulder=0 now unambiguously
                # means "upper arm vertical." However, WHICH direction
                # increasing shoulder_flexion_joint swings the arm
                # (forward vs. backward) depends on a world-frame
                # convention that can only be confirmed by looking at
                # the rendered scene -- left at 0.0 here deliberately to
                # avoid guessing on that, since 0.0 is unambiguous
                # regardless of that convention.
                "shoulder_flexion_joint": 0.0,
                "shoulder_abduction_joint": 0.0,
                "shoulder_rotation_joint": 0.0,
                # ~90 deg elbow bend -- brings the forearm away from
                # vertical, matching "elbow to wrist tilted forward."
                "elbow_flexion_joint": 1.57,
                "forearm_rotation_joint": 0.0,
                # Near the wrist's clinical flexion limit (80 deg max;
                # 1.3 rad = ~74.5 deg), representing a relaxed hand
                # drooping down under gravity from the now-horizontal
                # forearm.
                "wrist_flexion_joint": 1.3,
                "wrist_deviation_joint": 0.0,
            },
        ),
        # Isaac Lab's ArticulationCfg requires an explicit actuators
        # definition (unlike RigidObjectCfg) -- this is the layer that
        # can apply active PD position/velocity control on top of the
        # raw physics joints. We set stiffness=0.0 and damping=0.0 here
        # deliberately: this disables Isaac Lab's active actuator control
        # entirely, so the ONLY forces acting on each joint are gravity,
        # external contact (the robot), and the passive
        # damping/friction already defined per-joint in the URDF's
        # <dynamics> tags. This keeps the arm genuinely passive, as
        # intended -- it does not fight to hold a position the way an
        # actively-driven joint would.
        actuators={
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
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
