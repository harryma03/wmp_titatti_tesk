import numpy as np

def quat_rotate(q, v):
    """
    用四元数 q 旋转向量 v
    q = (w,x,y,z)
    """
    w, x, y, z = q

    # quaternion乘法
    def qmul(a, b):
        w1,x1,y1,z1 = a
        w2,x2,y2,z2 = b
        return (
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        )

    q_conj = (w, -x, -y, -z)
    v_q = (0, *v)

    return qmul(qmul(q, v_q), q_conj)[1:]


# =========================
# 测试你的quat
# =========================

# q = (0.5, 0.5, -0.5, -0.5)

# # MuJoCo forward = -Z
# forward = quat_rotate(q, (0, 0, -1))
# right   = quat_rotate(q, (1, 0, 0))
# up      = quat_rotate(q, (0, 1, 0))

# print("forward:", forward)
# print("right  :", right)
# print("up     :", up)


import numpy as np

def qmul(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    )

def axis_angle_to_quat(axis, angle_deg):
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    angle = np.deg2rad(angle_deg)
    c = np.cos(angle/2)
    s = np.sin(angle/2)

    return (c, axis[0]*s, axis[1]*s, axis[2]*s)

def rotate_local(q, axis, angle_deg):
    """
    在当前姿态 q 上，绕“局部轴”旋转
    axis: 'x' / 'y' / 'z'
    """
    if axis == 'x':
        ax = (1,0,0)
    elif axis == 'y':
        ax = (0,1,0)
    elif axis == 'z':
        ax = (0,0,1)
    else:
        raise ValueError("axis must be x/y/z")

    q_rot = axis_angle_to_quat(ax, angle_deg)

    return qmul(q, q_rot)   # 关键：右乘


q = (0.5, 0.5, -0.5, -0.5)
q_new = rotate_local(q, 'x', -10)
print(*q_new)