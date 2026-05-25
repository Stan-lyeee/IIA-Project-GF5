from __future__ import annotations

import numpy as np


def make_one_hot_skinning_weights(weights: np.ndarray) -> np.ndarray:
    """Student part-2 task: convert a dense weight matrix into one-hot weights."""
    weights = np.array(weights)
    one_hot_weights = np.zeros_like(weights)

    for vertex_index in range(len(weights)):
        max_index = np.argmax(weights[vertex_index])
        one_hot_weights[vertex_index][max_index] = 1.0

    return one_hot_weights


def skin_smpl_mesh(
    model_data: object,
    world_rotations: np.ndarray,
    world_positions: np.ndarray,
    *,
    use_blended_weights: bool,
) -> np.ndarray:
    """Student part-2 task: pose the SMPL mesh with one-hot or blended weights."""
    skinned_vertices = np.zeros_like(model_data.rest_vertices)
    if use_blended_weights:
        weights = model_data.skinning_weights
    else:
        weights = model_data.one_hot_skinning_weights

    for vertex_index in range(len(model_data.rest_vertices)):
        position = np.zeros(3)
        for joint_index in range(len(model_data.rest_joints)):
            weight = weights[vertex_index][joint_index]
            rest_offset = model_data.rest_vertices[vertex_index] - model_data.rest_joints[joint_index]
            transformed_position = world_positions[joint_index] + world_rotations[joint_index] @ rest_offset
            position += weight * transformed_position
        skinned_vertices[vertex_index] = position
    return skinned_vertices
