"""SMPL-X neutral body model restricted to the 22 body joints (TransPose-style interface).

Loads the official ``SMPLX_NEUTRAL.npz``. Joint order and parents of the first 22 SMPL-X joints
match SMPL joints 0-21 (pelvis ... right wrist). Hands, jaw, eyes and expressions are not
modelled: their rotations are identity for the mesh and they are excluded from joints.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

NUM_BODY_JOINTS = 22
NUM_BETAS = 10


def _forward_tree(x_local: torch.Tensor, parent: List[Optional[int]], op) -> torch.Tensor:
    """Compose ``x_local (B, J, ...)`` along the kinematic tree; ``op(parent_global, local)``."""
    x_global = [x_local[:, 0]]
    for i in range(1, len(parent)):
        x_global.append(op(x_global[parent[i]], x_local[:, i]))
    return torch.stack(x_global, dim=1)


def _inverse_tree(x_global: torch.Tensor, parent: List[Optional[int]], op, inverse) -> torch.Tensor:
    """Undo :func:`_forward_tree`; ``op(inverse(parent_global), global)``."""
    x_local = [x_global[:, 0]]
    for i in range(1, len(parent)):
        x_local.append(op(inverse(x_global[:, parent[i]]), x_global[:, i]))
    return torch.stack(x_local, dim=1)


def transformation_matrix(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Stack ``R (..., 3, 3)`` and ``p (..., 3)`` into homogeneous ``(..., 4, 4)`` matrices."""
    T = torch.zeros(R.shape[:-2] + (4, 4), dtype=R.dtype, device=R.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = p
    T[..., 3, 3] = 1.0
    return T


class SMPLXBodyModel:
    """Forward / inverse kinematics and linear-blend skinning for the SMPL-X body joints."""

    def __init__(self, model_path: Path, device: torch.device = torch.device("cpu"),
                 use_pose_blendshape: bool = True, skinning: bool = True):
        """``skinning=False`` skips the ~60 MB skinning / blend-shape tensors when only joints are needed."""
        data = np.load(str(model_path), allow_pickle=True)
        self.device = device
        self.use_pose_blendshape = use_pose_blendshape
        self.skinning = skinning
        self._v_template = torch.from_numpy(np.asarray(data["v_template"], dtype=np.float32)).to(device)  # (V, 3)
        self._J_regressor = torch.from_numpy(np.asarray(data["J_regressor"], dtype=np.float32)).to(device)  # (55, V)
        self._shapedirs = torch.from_numpy(
            np.asarray(data["shapedirs"], dtype=np.float32)[:, :, :NUM_BETAS]
        ).to(device)  # (V, 3, 10)
        self.face = np.asarray(data["f"], dtype=np.int64)
        self._skinning_weights = None
        self._posedirs = None
        if skinning:
            self._skinning_weights = torch.from_numpy(np.asarray(data["weights"], dtype=np.float32)).to(device)  # (V, 55)
            # (V, 3, 486): pose blend shapes for the 54 non-root joints, applied to the mesh only.
            self._posedirs = torch.from_numpy(np.asarray(data["posedirs"], dtype=np.float32)).to(device)

        kintree = np.asarray(data["kintree_table"])[0].astype(np.int64)  # root parent is uint32(-1)
        self.parent_full: List[Optional[int]] = [None] + kintree[1:].tolist()
        self.num_joints_full = len(self.parent_full)
        self.parent: List[Optional[int]] = self.parent_full[:NUM_BODY_JOINTS]
        assert all(p is not None and p < i for i, p in enumerate(self.parent) if i > 0), self.parent

        self._J_full = self._J_regressor @ self._v_template  # (55, 3) rest joints, template shape
        self._J = self._J_full[:NUM_BODY_JOINTS]  # (22, 3)
        self.num_vertices = self._v_template.shape[0]

    def get_zero_pose_joint_and_vertex(self, shape: Optional[torch.Tensor] = None):
        """Rest-pose body joints and vertices with the pelvis at the origin.

        Returns ``(J (22,3), V (num_vertices,3))`` for the template shape, or batched
        ``((B,22,3), (B,num_vertices,3))`` when ``shape (B,10)`` is given."""
        if shape is None:
            return self._J - self._J[:1], self._v_template - self._J[:1]
        shape = shape.view(-1, NUM_BETAS).to(self.device)
        v = torch.tensordot(shape, self._shapedirs, dims=([1], [2])) + self._v_template  # (B, V, 3)
        j = torch.matmul(self._J_regressor, v)[:, :NUM_BODY_JOINTS]  # (B, 22, 3)
        return j - j[:, :1], v - j[:, :1]

    def forward_kinematics_R(self, R_local: torch.Tensor) -> torch.Tensor:
        """Local ``(B,22,3,3)`` to global rotation matrices."""
        return _forward_tree(R_local, self.parent, torch.matmul)

    def inverse_kinematics_R(self, R_global: torch.Tensor) -> torch.Tensor:
        """Global ``(B,22,3,3)`` to local rotation matrices."""
        return _inverse_tree(R_global, self.parent, torch.matmul, lambda R: R.transpose(-1, -2))

    def forward_kinematics_T(self, T_local: torch.Tensor) -> torch.Tensor:
        """Local ``(B,J,4,4)`` to global transforms over the body tree (or full tree if ``J == 55``)."""
        parent = self.parent if T_local.shape[1] == NUM_BODY_JOINTS else self.parent_full
        return _forward_tree(T_local, parent, torch.matmul)

    def joint_position_to_bone_vector(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Joint positions ``(B,J,3)`` to parent-relative bone vectors."""
        parent = self.parent if joint_pos.shape[1] == NUM_BODY_JOINTS else self.parent_full
        return _inverse_tree(joint_pos, parent, torch.add, torch.neg)

    def bone_vector_to_joint_position(self, bone_vec: torch.Tensor) -> torch.Tensor:
        """Parent-relative bone vectors ``(B,J,3)`` to joint positions."""
        parent = self.parent if bone_vec.shape[1] == NUM_BODY_JOINTS else self.parent_full
        return _forward_tree(bone_vec, parent, torch.add)

    def forward_kinematics(
        self,
        pose: torch.Tensor,
        shape: Optional[torch.Tensor] = None,
        tran: Optional[torch.Tensor] = None,
        calc_mesh: bool = False,
    ):
        """Global joint rotations ``(B,22,3,3)``, joint positions ``(B,22,3)`` and optionally skinned
        vertices ``(B,num_vertices,3)`` from local rotations ``pose`` reshapeable to ``(B,22,3,3)``."""
        B = pose.shape[0]
        pose = pose.view(B, NUM_BODY_JOINTS, 3, 3)
        j, v = self.get_zero_pose_joint_and_vertex(shape)
        j = j.expand(B, -1, -1) if j.dim() == 2 else j  # (B, 22, 3)
        v = v.expand(B, -1, -1) if v.dim() == 2 else v  # (B, V, 3)
        offset = tran.view(-1, 1, 3) if tran is not None else torch.zeros(1, 1, 3, device=pose.device, dtype=pose.dtype)

        T_local = transformation_matrix(pose, self.joint_position_to_bone_vector(j))  # (B, 22, 4, 4)
        T_global = self.forward_kinematics_T(T_local)
        pose_global, joint_global = T_global[..., :3, :3].clone(), T_global[..., :3, 3].clone()
        if not calc_mesh:
            return pose_global, joint_global + offset
        assert self.skinning, "SMPLXBodyModel was created with skinning=False; cannot compute the mesh"

        # Skinning needs all 55 SMPL-X joints; non-body joints stay at their rest orientation.
        j_full = self._J_full.expand(B, -1, -1) - self._J_full[:1]  # (B, 55, 3), pelvis at origin
        if shape is not None:
            v_shaped = torch.tensordot(shape.view(-1, NUM_BETAS).to(self.device), self._shapedirs, dims=([1], [2])) + self._v_template
            j_full = torch.matmul(self._J_regressor, v_shaped)
            j_full = j_full - j_full[:, :1]
        R_full = torch.eye(3, device=pose.device, dtype=pose.dtype).expand(B, self.num_joints_full, 3, 3).clone()
        R_full[:, :NUM_BODY_JOINTS] = pose
        T_full = self.forward_kinematics_T(transformation_matrix(R_full, self.joint_position_to_bone_vector(j_full)))
        # T_full maps rest-pose points expressed relative to each joint; subtract the joint rest position.
        T_full[..., :3, 3] -= torch.matmul(T_full[..., :3, :3], j_full.unsqueeze(-1)).squeeze(-1)
        T_vertex = torch.einsum("bjmn,vj->bvmn", T_full, self._skinning_weights)  # (B, V, 4, 4)
        if self.use_pose_blendshape:
            eye = torch.eye(3, device=pose.device, dtype=pose.dtype)
            r = (R_full[:, 1:] - eye).flatten(1)  # (B, 486)
            v = v + torch.tensordot(r, self._posedirs, dims=([1], [2]))  # (B, V, 3)
        v_h = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)  # (B, V, 4)
        vertex_global = torch.matmul(T_vertex, v_h.unsqueeze(-1)).squeeze(-1)[..., :3]
        return pose_global, joint_global + offset, vertex_global + offset
