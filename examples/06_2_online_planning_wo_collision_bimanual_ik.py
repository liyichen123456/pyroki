"""Online Planning

Run online planning without collision, but with two end effectors!
"""

import time

import numpy as np
import pyroki as pk
import viser
from viser.extras import ViserUrdf
import yourdfpy
import pyroki_snippets as pks
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
DIR_PATH = FILE_PATH.parent.parent.parent.parent


def main():
    """Main function for online planning with collision."""
    # openarm_bimanual
    urdf = yourdfpy.URDF.load(
        f"{DIR_PATH}/thirdparty/openarm_wuji_description/openarm_wuji_bimanual.urdf",
        mesh_dir=f"{DIR_PATH}/thirdparty/openarm_wuji_description",
    )
    target_link_names = ["right_palm_link", "left_palm_link"]
    robot = pk.Robot.from_urdf(urdf)

    # Define the online planning parameters.
    len_traj, dt = 5, 0.02

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    robot_base = server.scene.add_frame("/robot")
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    base_offset = np.array([0, 0, 0.0])
    robot_base.position = base_offset

    # Create interactive controller for IK target.
    ik_target_0 = server.scene.add_transform_controls(
        "/ik_target_0", scale=0.2, position=(0.41, -0.3, 0.56), wxyz=(0.501, 0.499, 0.497, 0.507)
    )
    ik_target_1 = server.scene.add_transform_controls(
        "/ik_target_1", scale=0.2, position=(0.41, 0.3, 0.56), wxyz=(0.506, -0.497, 0.505, -0.502)
    )

    # target_frame_handle = server.scene.add_batched_axes(
    #     "target_frame",
    #     axes_length=0.05,
    #     axes_radius=0.005,
    #     batched_positions=np.zeros((25, 3)),
    #     batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]] * 25),
    # )

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    sol_pos, sol_wxyz = None, None
    sol_traj = np.array(
        robot.joint_var_cls.default_factory()[None].repeat(len_traj, axis=0)
    )
    while True:
        start_time = time.time()

        sol_traj, sol_pos, sol_wxyz = pks.solve_online_planning_with_multi_targets_wo_collision(
            robot=robot,
            target_link_names=target_link_names,
            target_positions=np.array([ik_target_0.position, ik_target_1.position]),
            target_wxyzs=np.array([ik_target_0.wxyz, ik_target_1.wxyz]),
            timesteps=len_traj,
            dt=dt,
            start_cfg=sol_traj[0],
            prev_sols=sol_traj,
        )

        # Update timing handle.
        timing_handle.value = (
            0.99 * timing_handle.value + 0.01 * (time.time() - start_time) * 1000
        )

        # Update visualizer.
        urdf_vis.update_cfg(
            sol_traj[-1]
        )  # The last step of the online trajectory solution.

        # Update the planned trajectory visualization.
        # if hasattr(target_frame_handle, "batched_positions"):
        #     target_frame_handle.batched_positions = np.array(sol_pos)  # type: ignore[attr-defined]
        #     target_frame_handle.batched_wxyzs = np.array(sol_wxyz)  # type: ignore[attr-defined]
        # else:
        #     # This is an older version of Viser.
        #     target_frame_handle.positions_batched = np.array(sol_pos)  # type: ignore[attr-defined]
        #     target_frame_handle.wxyzs_batched = np.array(sol_wxyz)  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
