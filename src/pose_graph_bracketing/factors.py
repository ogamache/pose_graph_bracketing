"""Custom GTSAM factors: constant-body-velocity motion prior, and a stereo
landmark-observation factor (thin robust-kernel wrapper around GTSAM's
built-in GenericStereoFactor3D).

The motion-prior factor uses central-difference numerical Jacobians (computed
via each variable's manifold `retract`), since GTSAM's Python bindings do not
expose automatic differentiation for CustomFactor error functions. This is
fine performance-wise here: it only touches Pose3 (dim 6) / Vector6 (dim 6)
variables, so at most ~24 error evaluations per factor per optimizer
iteration.
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


def make_stereo_observation_factor(
    pose_key: int,
    landmark_key: int,
    stereo_point: np.ndarray,
    K_stereo: gtsam.Cal3_S2Stereo,
    pixel_sigma: float,
    huber_k: float,
) -> gtsam.GenericStereoFactor3D:
    """A single rectified-stereo reprojection observation of a persistent landmark.

    `stereo_point` is `[uL, uR, v]` in rectified pixel coordinates (see
    stereo.compute_stereo_observations). Ties `pose_key`'s camera to
    `landmark_key`'s 3D position via GTSAM's stereo camera model -- unlike the
    earlier pairwise PnP-derived BetweenFactorPose3 VO factor, this lets the
    same landmark accumulate observations from every frame that sees it,
    giving the optimizer real multi-view geometric redundancy instead of a
    fresh one-shot pose estimate per frame pair.

    The noise model is wrapped in a Huber robust kernel: a single bad
    observation (e.g. a mismatch during a fast-rotation segment) gets
    down-weighted in the optimization instead of directly corrupting the
    landmark/pose estimate the way a plain least-squares residual would.
    """
    measured = gtsam.StereoPoint2(float(stereo_point[0]), float(stereo_point[1]), float(stereo_point[2]))
    base_noise = gtsam.noiseModel.Isotropic.Sigma(3, pixel_sigma)
    robust_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(huber_k), base_noise)
    return gtsam.GenericStereoFactor3D(measured, robust_noise, pose_key, landmark_key, K_stereo)
