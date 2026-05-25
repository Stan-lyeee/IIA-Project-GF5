from __future__ import annotations

import numpy as np


def forward_kinematics(
    joints: list[object],
    local_rotations: list[np.ndarray],
    root_offset: np.ndarray,
    topological_order: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Student part-1 implementation.

    Expected inputs:
    - joints: each joint has `.parent` and `.translation`
    - local_rotations: one 3x3 local rotation matrix per joint
    - root_offset: global translation applied to the root
    - topological_order: optional parent-before-child traversal order

    Expected outputs:
    - world_rotations: shape (J, 3, 3)
    - world_positions: shape (J, 3)
    """

    joint_count = len(joints)
    world_rotations = np.zeros((joint_count, 3, 3))
    world_positions = np.zeros((joint_count, 3))
    
    def make_topological_order(joints):
        order = []
        visited = [False] * joint_count

        def dfs(index):
            if visited[index]:
                return
            visited[index] = True

            parent = joints[index].parent
            if parent >= 0:
                dfs(parent)
            order.append(index)
        
        for index in range(joint_count):
            dfs(index)
        
        return order

    if topological_order is None:
        iteration_order = make_topological_order(joints)
    else:
        iteration_order = topological_order

    for index in iteration_order:
        joint_parent = joints[index].parent
        if joint_parent < 0:
            world_positions[index] = joints[index].translation + root_offset
            world_rotations[index] = local_rotations[index]
        else:
            world_rotations[index] = world_rotations[joint_parent] @ local_rotations[index]
            world_positions[index] = world_positions[joint_parent] + world_rotations[joint_parent] @ joints[index].translation


    return world_rotations, world_positions


