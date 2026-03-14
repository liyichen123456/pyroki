from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


def solve_online_planning(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    target_position: onp.ndarray,
    target_wxyz: onp.ndarray,
    timesteps: int,
    dt: float,
    start_cfg: onp.ndarray,
    prev_sols: onp.ndarray,
) -> tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """Solve online planning with collision."""

    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_indices = [robot.links.names.index(target_link_name)]

    target_poses = jaxlie.SE3(
        jnp.concatenate([jnp.array(target_wxyz), jnp.array(target_position)], axis=-1)
    )
    target_links = jnp.array(target_link_indices)

    # Warm start: use previous solution shifted by one step.
    timesteps = timesteps + 1  # for start pose cost.

    sol_traj, sol_pos, sol_wxyz = _solve_online_planning_jax(
        robot,
        robot_coll,
        world_coll,
        target_poses,
        target_links,
        timesteps,
        dt,
        jnp.array(start_cfg),
        jnp.concatenate([prev_sols, prev_sols[-1:]], axis=0),
    )
    sol_traj = sol_traj[1:]
    sol_pos = sol_pos[1:]
    sol_wxyz = sol_wxyz[1:]

    return onp.array(sol_traj), onp.array(sol_pos), onp.array(sol_wxyz)


@jdc.jit
def _solve_online_planning_jax(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_poses: jaxlie.SE3,
    target_links: jnp.ndarray,
    timesteps: jdc.Static[int],
    dt: float,
    start_cfg: jnp.ndarray,
    prev_sols: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    num_targets = len(target_links)

    def batched_rplus(
        pose: jaxlie.SE3,
        delta: jax.Array,
    ) -> jaxlie.SE3:
        return jax.vmap(jaxlie.manifold.rplus)(pose, delta.reshape(num_targets, -1))

    # Custom SE3 variable to batch across multiple joint targets.
    # This is not to be confused with SE3Vars with ids, which we use here for timesteps.
    class BatchedSE3Var(  # pylint: disable=missing-class-docstring
        jaxls.Var[jaxlie.SE3],
        default_factory=lambda: jaxlie.SE3.identity((num_targets,)),
        retract_fn=batched_rplus,
        tangent_dim=jaxlie.SE3.tangent_dim * num_targets,
    ): ...

    # --- Define Variables ---
    traj_var = robot.joint_var_cls(jnp.arange(0, timesteps))
    traj_var_prev = robot.joint_var_cls(jnp.arange(0, timesteps - 1))
    traj_var_next = robot.joint_var_cls(jnp.arange(1, timesteps))
    pose_var = BatchedSE3Var(jnp.arange(0, timesteps))
    pose_var_prev = BatchedSE3Var(jnp.arange(0, timesteps - 1))
    pose_var_next = BatchedSE3Var(jnp.arange(1, timesteps))

    init_pose_vals = jaxlie.SE3(
        robot.forward_kinematics(prev_sols)[..., target_links, :]
    )

    # --- Define Costs ---
    factors: list[jaxls.Cost] = []  # Changed type hint to jaxls.Cost

    @jaxls.Cost.factory(name="SE3PoseMatchJointCost")
    def match_joint_to_pose_cost(
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[jnp.ndarray],
        pose_var: BatchedSE3Var,
    ):
        joint_cfg = vals[joint_var]
        target_pose = vals[pose_var]
        Ts_joint_world = robot.forward_kinematics(joint_cfg)
        residual = (
            (jaxlie.SE3(Ts_joint_world[..., target_links, :])).inverse() @ (target_pose)
        ).log()
        return residual.flatten() * 100.0

    @jaxls.Cost.factory(name="SE3SmoothnessCost")
    def pose_smoothness_cost(
        vals: jaxls.VarValues,
        pose_var: BatchedSE3Var,
        pose_var_prev: BatchedSE3Var,
    ):
        return (vals[pose_var].inverse() @ vals[pose_var_prev]).log().flatten() * 1.0

    @jaxls.Cost.factory(name="SE3PoseMatchCost")
    def pose_match_cost(
        vals: jaxls.VarValues,
        pose_var: BatchedSE3Var,
    ):
        return (
            (vals[pose_var].inverse() @ target_poses).log()
            * jnp.array([50.0] * 3 + [20.0] * 3)
        ).flatten()

    @jaxls.Cost.factory(name="MatchStartPoseCost")
    def match_start_pose_cost(
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[jnp.ndarray],
    ):
        return (vals[joint_var] - start_cfg).flatten() * 100.0

    # Add pose costs.
    factors.extend(
        [
            pose_match_cost(
                BatchedSE3Var(timesteps - 1),
            ),
            pose_smoothness_cost(
                pose_var_next,
                pose_var_prev,
            ),
        ]
    )

    # Need to constrain the start joint cfg.
    factors.append(match_start_pose_cost(robot.joint_var_cls(0)))

    # Add joint costs.
    # Technically we should add this as a constraint, but the augmented lagrangian formulation
    # requires us to perform multiple outer loop iterations to converge, which is slow. :(
    factors.extend(
        [
            match_joint_to_pose_cost(
                traj_var,
                pose_var,
            ),
            pk.costs.smoothness_cost(
                traj_var_prev,
                traj_var_next,
                weight=10.0,
            ),
            pk.costs.limit_velocity_cost(
                jax.tree.map(lambda x: x[None], robot),
                traj_var_prev,
                traj_var_next,
                weight=1.0,
                dt=dt,
            ),
            pk.costs.limit_cost(
                jax.tree.map(lambda x: x[None], robot),
                traj_var,
                weight=100.0,
            ),
            pk.costs.rest_cost(
                traj_var,
                jnp.array(traj_var.default_factory())[None],
                weight=0.01,
            ),
            pk.costs.manipulability_cost(
                jax.tree.map(lambda x: x[None], robot),
                traj_var,
                weight=0.01,
                target_link_indices=target_links,
            ),
            pk.costs.self_collision_cost(
                jax.tree.map(lambda x: x[None], robot),
                jax.tree.map(lambda x: x[None], robot_coll),
                traj_var,
                weight=10.0,
                margin=0.02,
            ),
        ]
    )
    factors.extend(
        [
            pk.costs.world_collision_cost(
                jax.tree.map(lambda x: x[None], robot),
                jax.tree.map(lambda x: x[None], robot_coll),
                traj_var,
                jax.tree.map(lambda x: x[None], obs),
                weight=20.0,
                margin=0.1,
            )
            for obs in world_coll
        ]
    )

    solution = (
        jaxls.LeastSquaresProblem(factors, [traj_var, pose_var])
        .analyze()
        .solve(
            verbose=False,
            initial_vals=jaxls.VarValues.make(
                (traj_var.with_value(prev_sols), pose_var.with_value(init_pose_vals))
            ),
            termination=jaxls.TerminationConfig(max_iterations=20),
        )
    )
    pose_traj = solution[pose_var]
    return (
        solution[traj_var],
        pose_traj.translation(),
        pose_traj.rotation().wxyz,
    )


def solve_online_planning_wo_collision(
    robot: pk.Robot,
    target_link_name: str,
    target_position: onp.ndarray,
    target_wxyz: onp.ndarray,
    timesteps: int,
    dt: float,
    start_cfg: onp.ndarray,
    prev_sols: onp.ndarray,
) -> tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """Online planning without collision, with teleop-friendly smoothness terms."""

    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_index = robot.links.names.index(target_link_name)

    # 和原实现一样：多加一帧用于 start pose cost
    timesteps = timesteps + 1

    sol_traj, sol_pos, sol_wxyz = _solve_online_planning_no_collision_jax(
        robot=robot,
        target_link_index=target_link_index,
        target_position=jnp.array(target_position),
        target_wxyz=jnp.array(target_wxyz),
        timesteps=timesteps,
        dt=dt,
        start_cfg=jnp.array(start_cfg),
        prev_sols=jnp.concatenate([jnp.array(prev_sols), jnp.array(prev_sols[-1:])], axis=0),
    )

    # 去掉第 0 帧（对应 start pose）
    sol_traj = sol_traj[1:]
    sol_pos = sol_pos[1:]
    sol_wxyz = sol_wxyz[1:]

    return onp.array(sol_traj), onp.array(sol_pos), onp.array(sol_wxyz)


@jdc.jit
def _solve_online_planning_no_collision_jax(
    robot: pk.Robot,
    target_link_index: jdc.Static[int],
    target_position: jnp.ndarray,
    target_wxyz: jnp.ndarray,
    timesteps: jdc.Static[int],
    dt: float,
    start_cfg: jnp.ndarray,
    prev_sols: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX solver for online planning without collision."""

    target_pose = jaxlie.SE3(
        jnp.concatenate([target_wxyz, target_position], axis=-1)
    )

    # trajectory variables
    traj_var = robot.joint_var_cls(jnp.arange(0, timesteps))
    traj_var_prev = robot.joint_var_cls(jnp.arange(0, timesteps - 1))
    traj_var_next = robot.joint_var_cls(jnp.arange(1, timesteps))

    factors: list[jaxls.Cost] = []

    # 1) 末端 tracking：只约束最后一帧接近 target
    @jaxls.Cost.factory(name="TrackingCost")
    def tracking_cost(
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[jnp.ndarray],
    ):
        q = vals[joint_var]
        Ts_joint_world = robot.forward_kinematics(q)
        T_ee = jaxlie.SE3(Ts_joint_world[..., target_link_index, :])
        residual = (T_ee.inverse() @ target_pose).log()
        return residual * jnp.array([80.0, 80.0, 80.0, 20.0, 20.0, 20.0])

    # 2) 起点约束：第一帧贴近当前真实配置
    @jaxls.Cost.factory(name="MatchStartCfgCost")
    def match_start_cfg_cost(
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[jnp.ndarray],
    ):
        return (vals[joint_var] - start_cfg).flatten() * 100.0

    # 3) velocity damping：显式惩罚速度
    @jaxls.Cost.factory(name="VelocityDampingCost")
    def velocity_damping_cost(
        vals: jaxls.VarValues,
        joint_var_prev: jaxls.Var[jnp.ndarray],
        joint_var_next: jaxls.Var[jnp.ndarray],
    ):
        q_prev = vals[joint_var_prev]
        q_next = vals[joint_var_next]
        vel = (q_next - q_prev) / dt
        return vel.flatten() * 0.5

    # add costs
    factors.append(tracking_cost(robot.joint_var_cls(timesteps - 1)))
    factors.append(match_start_cfg_cost(robot.joint_var_cls(0)))

    factors.extend(
        [
            # joint smoothness
            pk.costs.smoothness_cost(
                traj_var_prev,
                traj_var_next,
                weight=20.0,
            ),
            # explicit velocity damping
            velocity_damping_cost(
                traj_var_prev,
                traj_var_next,
            ),
            # joint velocity limit
            pk.costs.limit_velocity_cost(
                jax.tree.map(lambda x: x[None], robot),
                traj_var_prev,
                traj_var_next,
                weight=2.0,
                dt=dt,
            ),
            # joint limit
            pk.costs.limit_cost(
                jax.tree.map(lambda x: x[None], robot),
                traj_var,
                weight=100.0,
            ),
            # mild regularization to rest pose
            pk.costs.rest_cost(
                traj_var,
                jnp.array(traj_var.default_factory())[None],
                weight=0.01,
            ),
            # optional: keep manipulability
            pk.costs.manipulability_cost(
                jax.tree.map(lambda x: x[None], robot),
                traj_var,
                weight=0.01,
                target_link_indices=jnp.array([target_link_index]),
            ),
        ]
    )

    solution = (
        jaxls.LeastSquaresProblem(factors, [traj_var])
        .analyze()
        .solve(
            verbose=False,
            initial_vals=jaxls.VarValues.make(
                (traj_var.with_value(prev_sols),)
            ),
            termination=jaxls.TerminationConfig(max_iterations=20),
        )
    )

    sol_traj = solution[traj_var]  # [T, dof]

    # 用 FK 计算每一帧末端位姿，保持和原接口一致
    Ts_all = robot.forward_kinematics(sol_traj)  # [T, num_links, 7]
    ee_traj = jaxlie.SE3(Ts_all[..., target_link_index, :])  # [T]

    sol_pos = ee_traj.translation()          # [T, 3]
    sol_wxyz = ee_traj.rotation().wxyz       # [T, 4]

    return sol_traj, sol_pos, sol_wxyz
