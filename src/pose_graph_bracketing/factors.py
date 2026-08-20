"""Custom GTSAM factors: constant-body-velocity motion prior and a
rotation + translation-direction-only visual-odometry factor.

Both factors use central-difference numerical Jacobians (computed via each
variable's manifold `retract`), since GTSAM's Python bindings do not expose
automatic differentiation for CustomFactor error functions. This is fine
performance-wise here: each factor touches only Pose3 (dim 6) / Vector6
(dim 6) variables, so at most ~24 error evaluations per factor per
optimizer iteration.
"""

from __future__ import annotations

import gtsam
import numpy as np

_EPS = 1e-6


def _retract(value, delta: np.ndarray):
    if isinstance(value, gtsam.Pose3):
        return value.retract(delta)
    return value + delta  # Vector-valued (numpy array)


def _numerical_jacobians(error_fn, values: list, dims: list[int]) -> list[np.ndarray]:
    """error_fn(values) -> np.ndarray; returns d(error)/d(local coords) per value."""
    base_err = error_fn(values)
    m = base_err.shape[0]
    jacobians = []
    for idx, (val, d) in enumerate(zip(values, dims)):
        J = np.zeros((m, d))
        for k in range(d):
            delta = np.zeros(d)
            delta[k] = _EPS
            perturbed = list(values)
            perturbed[idx] = _retract(val, delta)
            err_plus = error_fn(perturbed)
            delta[k] = -_EPS
            perturbed[idx] = _retract(val, delta)
            err_minus = error_fn(perturbed)
            J[:, k] = (err_plus - err_minus) / (2 * _EPS)
        jacobians.append(J)
    return jacobians


def predict_pose(pose_i: gtsam.Pose3, vel_i: np.ndarray, dt: float) -> gtsam.Pose3:
    """Constant body-velocity kinematic prediction.

    vel_i = [wx, wy, wz, vx, vy, vz], body-frame angular + linear velocity.
    """
    w = vel_i[0:3]
    v = vel_i[3:6]
    step = gtsam.Pose3(gtsam.Rot3.Expmap(w * dt), v * dt)
    return pose_i.compose(step)


def motion_prior_noise_model(
    rotation_sigma: float,
    translation_sigma: float,
    angular_velocity_rw_sigma: float,
    linear_velocity_rw_sigma: float,
    dt: float,
) -> gtsam.noiseModel.Base:
    dt_sqrt = max(np.sqrt(max(dt, 1e-6)), 1e-6)
    sigmas = np.array(
        [
            rotation_sigma * dt_sqrt,
            rotation_sigma * dt_sqrt,
            rotation_sigma * dt_sqrt,
            translation_sigma * dt_sqrt,
            translation_sigma * dt_sqrt,
            translation_sigma * dt_sqrt,
            angular_velocity_rw_sigma * dt_sqrt,
            angular_velocity_rw_sigma * dt_sqrt,
            angular_velocity_rw_sigma * dt_sqrt,
            linear_velocity_rw_sigma * dt_sqrt,
            linear_velocity_rw_sigma * dt_sqrt,
            linear_velocity_rw_sigma * dt_sqrt,
        ]
    )
    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)


def make_motion_prior_factor(
    key_pose_i: int,
    key_vel_i: int,
    key_pose_j: int,
    key_vel_j: int,
    dt: float,
    noise_model: gtsam.noiseModel.Base,
) -> gtsam.CustomFactor:
    """Constant-body-velocity motion prior between chronologically consecutive frames.

    Residual (12,): [Logmap(predicted(X_i, V_i, dt).between(X_j)); V_j - V_i]
    """

    def raw_error(vals: list) -> np.ndarray:
        pose_i, vel_i, pose_j, vel_j = vals
        predicted = predict_pose(pose_i, vel_i, dt)
        pose_err = gtsam.Pose3.Logmap(predicted.between(pose_j))
        vel_err = vel_j - vel_i
        return np.concatenate([pose_err, vel_err])

    def error_func(this: gtsam.CustomFactor, values: gtsam.Values, H: list | None) -> np.ndarray:
        vals = [
            values.atPose3(key_pose_i),
            values.atVector(key_vel_i),
            values.atPose3(key_pose_j),
            values.atVector(key_vel_j),
        ]
        err = raw_error(vals)
        if H is not None:
            jacobians = _numerical_jacobians(raw_error, vals, [6, 6, 6, 6])
            for i, J in enumerate(jacobians):
                H[i] = J
        return err

    return gtsam.CustomFactor(
        noise_model,
        [key_pose_i, key_vel_i, key_pose_j, key_vel_j],
        error_func,
    )


def vo_noise_model(rotation_sigma: float, translation_direction_sigma: float) -> gtsam.noiseModel.Base:
    sigmas = np.array(
        [
            rotation_sigma,
            rotation_sigma,
            rotation_sigma,
            translation_direction_sigma,
            translation_direction_sigma,
            translation_direction_sigma,
        ]
    )
    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)


def make_vo_direction_factor(
    key_pose_i: int,
    key_pose_j: int,
    R_ij: np.ndarray,
    t_hat_ij: np.ndarray,
    noise_model: gtsam.noiseModel.Base,
) -> gtsam.CustomFactor:
    """Visual-odometry factor: full rotation constraint, translation-*direction*-only.

    `R_ij`, `t_hat_ij` are the VO-estimated relative rotation and unit
    translation direction taking points from camera i into camera j
    (p_j ~ R_ij @ p_i + t_ij, scale unknown). Monocular VO cannot recover
    metric scale, so only the translation *direction* is constrained,
    leaving the graph's overall scale to be resolved (weakly) by the
    motion-prior velocity states.

    Residual (6,): [Logmap(R_ij^-1 * R_i^-1 * R_j); unit(t_rel) - t_hat_ij]
    where t_rel is the relative translation of X_j w.r.t. X_i.
    """
    R_vo = gtsam.Rot3(R_ij)
    t_hat = t_hat_ij / (np.linalg.norm(t_hat_ij) + 1e-12)

    def raw_error(vals: list) -> np.ndarray:
        pose_i, pose_j = vals
        rel = pose_i.between(pose_j)
        rot_err = gtsam.Rot3.Logmap(R_vo.inverse().compose(rel.rotation()))
        t_rel = rel.translation()
        t_rel_norm = t_rel / (np.linalg.norm(t_rel) + 1e-12)
        dir_err = t_rel_norm - t_hat
        return np.concatenate([rot_err, dir_err])

    def error_func(this: gtsam.CustomFactor, values: gtsam.Values, H: list | None) -> np.ndarray:
        vals = [values.atPose3(key_pose_i), values.atPose3(key_pose_j)]
        err = raw_error(vals)
        if H is not None:
            jacobians = _numerical_jacobians(raw_error, vals, [6, 6])
            for i, J in enumerate(jacobians):
                H[i] = J
        return err

    return gtsam.CustomFactor(noise_model, [key_pose_i, key_pose_j], error_func)


def make_metric_vo_factor(
    key_pose_i: int,
    key_pose_j: int,
    R_ij: np.ndarray,
    t_ij: np.ndarray,
    rotation_sigma: float,
    translation_sigma: float,
) -> gtsam.BetweenFactorPose3:
    """Metric stereo visual-odometry factor.

    `R_ij`, `t_ij` come from stereo-triangulation + PnP (odometry.estimate_pose_pnp):
    since the 3D points are triangulated using the known stereo baseline,
    `t_ij` is already a genuine metric translation (meters) -- no scale
    ambiguity/bootstrapping needed, unlike monocular essential-matrix VO.
    """
    measured = gtsam.Pose3(gtsam.Rot3(R_ij), t_ij)
    sigmas = np.array([rotation_sigma] * 3 + [translation_sigma] * 3)
    noise_model = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
    return gtsam.BetweenFactorPose3(key_pose_i, key_pose_j, measured, noise_model)
