"""
Solves the basic IK problem.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


def solve_ik_with_multiple_targets(
    robot: pk.Robot,
    target_link_names: Sequence[str],
    target_wxyzs: onp.ndarray,
    target_positions: onp.ndarray,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_names: Sequence[str]. List of link names to be controlled.
        target_wxyzs: onp.ndarray. Shape: (num_targets, 4). Target orientations.
        target_positions: onp.ndarray. Shape: (num_targets, 3). Target positions.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    num_targets = len(target_link_names)
    assert target_positions.shape == (num_targets, 3)
    assert target_wxyzs.shape == (num_targets, 4)
    target_link_indices = [robot.links.names.index(name) for name in target_link_names]

    cfg = _solve_ik_jax(
        robot,
        jnp.array(target_wxyzs),
        jnp.array(target_positions),
        jnp.array(target_link_indices),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    return onp.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    target_joint_indices: jax.Array,
) -> jax.Array:
    JointVar = robot.joint_var_cls

    # Get the batch axes for the variable through the target pose.
    # Batch axes for the variables and cost terms (e.g., target pose) should be broadcastable!
    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_position
    )
    batch_axes = target_pose.get_batch_axes()

    costs = [
        pk.costs.pose_cost_analytic_jac(
            jax.tree.map(lambda x: x[None], robot),
            JointVar(jnp.full(batch_axes, 0)),
            target_pose,
            target_joint_indices,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.rest_cost(
            JointVar(0),
            rest_pose=JointVar.default_factory(),
            weight=1.0,
        ),
    ]
    costs.append(
        pk.costs.limit_constraint(
            robot,
            JointVar(0),
        ),
    )
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[JointVar(0)])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
        )
    )
    return sol[JointVar(0)]


def solve_online_planning_with_multi_targets_wo_collision(
    robot: pk.Robot,
    target_link_names: Sequence[str],
    target_positions: onp.ndarray,  # (num_targets, 3)
    target_wxyzs: onp.ndarray,    # (num_targets, 4)
    timesteps: int,
    dt: float,
    start_cfg: onp.ndarray,
    prev_sols: onp.ndarray,
):
    """支持多目标的在线规划封装函数"""

    num_targets = len(target_link_names)
    assert target_positions.shape == (num_targets, 3)
    assert target_wxyzs.shape == (num_targets, 4)

    # 将 Link 名称转换为索引
    target_link_indices = [robot.links.names.index(name) for name in target_link_names]

    # 增加一帧用于 start pose cost 逻辑
    timesteps_internal = timesteps + 1

    sol_traj, sol_pos, sol_wxyz = _solve_online_planning_with_multi_targets_wo_collision_jax(
        robot=robot,
        target_link_indices=jnp.array(target_link_indices),
        target_positions=jnp.array(target_positions),
        target_wxyzs=jnp.array(target_wxyzs),
        timesteps=timesteps_internal,
        dt=dt,
        start_cfg=jnp.array(start_cfg),
        prev_sols=jnp.concatenate([jnp.array(prev_sols), jnp.array(prev_sols[-1:])], axis=0),
    )

    # 去掉用于约束起点的第 0 帧
    return onp.array(sol_traj[1:]), onp.array(sol_pos[1:]), onp.array(sol_wxyz[1:])


@jdc.jit
def _solve_online_planning_with_multi_targets_wo_collision_jax(
    robot: pk.Robot,
    target_link_indices: jnp.ndarray,
    target_positions: jnp.ndarray,
    target_wxyzs: jnp.ndarray,
    timesteps: jdc.Static[int],
    dt: float,
    start_cfg: jnp.ndarray,
    prev_sols: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    target_poses = jax.vmap(
        lambda w, p: jaxlie.SE3(jnp.concatenate([w, p], axis=-1))
    )(target_wxyzs, target_positions)

    traj_var = robot.joint_var_cls(jnp.arange(0, timesteps))
    traj_var_prev = robot.joint_var_cls(jnp.arange(0, timesteps - 1))
    traj_var_next = robot.joint_var_cls(jnp.arange(1, timesteps))

    factors: list[jaxls.Cost] = []

    @jaxls.Cost.factory(name="MultiTrackingCost")
    def multi_tracking_cost(vals: jaxls.VarValues, joint_var: jaxls.Var[jnp.ndarray]):
        q = vals[joint_var]
        Ts_joint_world = robot.forward_kinematics(q)  # (num_links, 7)
        current_ees_raw = Ts_joint_world[target_link_indices, :]  # (num_targets, 7)

        def compute_single_residual(curr_raw, target_pose):
            curr_pose = jaxlie.SE3(curr_raw)
            return (curr_pose.inverse() @ target_pose).log()

        residuals = jax.vmap(compute_single_residual)(current_ees_raw, target_poses)

        # SE3.log() 一般是 [tx, ty, tz, rx, ry, rz]
        weights = jnp.array([80.0, 80.0, 80.0, 20.0, 20.0, 20.0])
        # weights = jnp.array([100.0, 100.0, 100.0, 30.0, 30.0, 30.0])
        return (residuals * weights).flatten()

    @jaxls.Cost.factory(name="MatchStartCfgCost")
    def match_start_cfg_cost(vals: jaxls.VarValues, joint_var: jaxls.Var[jnp.ndarray]):
        return (vals[joint_var] - start_cfg).flatten() * 100.0  # 20

    @jaxls.Cost.factory(name="VelocityDampingCost")
    def velocity_damping_cost(
        vals: jaxls.VarValues,
        joint_var_prev: jaxls.Var[jnp.ndarray],
        joint_var_next: jaxls.Var[jnp.ndarray],
    ):
        vel = (vals[joint_var_next] - vals[joint_var_prev]) / dt
        return vel.flatten() * 0.5  # 0.05

    # 起点约束
    factors.append(match_start_cfg_cost(robot.joint_var_cls(0)))

    # 对每一帧都加 tracking，避免只有最后一帧动
    for t in range(timesteps):
        factors.append(multi_tracking_cost(robot.joint_var_cls(t)))

    factors.extend([
        pk.costs.smoothness_cost(
            traj_var_prev,
            traj_var_next,
            weight=20.0,
            # weight=5.0,
        ),
        velocity_damping_cost(
            traj_var_prev,
            traj_var_next,
        ),
        pk.costs.limit_velocity_cost(
            jax.tree.map(lambda x: x[None], robot),
            traj_var_prev,
            traj_var_next,
            weight=2.0,
            # weight=0.5,
            dt=dt,
        ),
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
    ])

    solution = (
        jaxls.LeastSquaresProblem(factors, [traj_var])
        .analyze()
        .solve(
            verbose=False,
            initial_vals=jaxls.VarValues.make((traj_var.with_value(prev_sols),)),
            termination=jaxls.TerminationConfig(max_iterations=20),
        )
    )

    sol_traj = solution[traj_var]
    Ts_all = robot.forward_kinematics(sol_traj)
    target_ees_traj = Ts_all[:, target_link_indices, :]

    sol_pos = target_ees_traj[..., 4:]
    sol_wxyz = target_ees_traj[..., :4]

    return sol_traj, sol_pos, sol_wxyz
