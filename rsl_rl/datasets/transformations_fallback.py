"""Small fallback for the pybullet_utils.transformations quaternion helpers."""

import math

import numpy as np


def _unit_vector(data):
    data = np.array(data, dtype=np.float64, copy=True)
    norm = np.linalg.norm(data)
    if norm < np.finfo(float).eps:
        return data
    return data / norm


def quaternion_about_axis(angle, axis):
    quat = np.zeros(4, dtype=np.float64)
    quat[:3] = _unit_vector(axis) * math.sin(angle / 2.0)
    quat[3] = math.cos(angle / 2.0)
    return quat


def quaternion_inverse(quaternion):
    quat = np.array(quaternion, dtype=np.float64, copy=True)
    quat[:3] *= -1.0
    return quat / np.dot(quat, quat)


def quaternion_multiply(quaternion1, quaternion0):
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array([
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
    ], dtype=np.float64)


def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    q0 = _unit_vector(quat0)
    q1 = _unit_vector(quat1)
    fraction = float(fraction)

    dot = np.dot(q0, q1)
    if shortestpath and dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)
    if abs(dot) > 0.999999:
        return _unit_vector(q0 + fraction * (q1 - q0))

    angle = math.acos(dot) + spin * math.pi
    sin_angle = math.sin(angle)
    weight0 = math.sin((1.0 - fraction) * angle) / sin_angle
    weight1 = math.sin(fraction * angle) / sin_angle
    return weight0 * q0 + weight1 * q1
