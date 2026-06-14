import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ═══════════════════════════════════════════════════════
#  UR5e — FK + IK + Animation
#  Built from scratch — no external robotics libraries!
# ═══════════════════════════════════════════════════════

# ── Joint Limits (radians) ──────────────────────────────
JOINT_LIMITS = [
    (-2*np.pi,  2*np.pi),   # shoulder_pan
    (-np.pi,    0),          # shoulder_lift
    (-np.pi,    np.pi),      # elbow
    (-2*np.pi,  2*np.pi),   # wrist_1
    (-2*np.pi,  2*np.pi),   # wrist_2
    (-2*np.pi,  2*np.pi),   # wrist_3
]

# ── STEP 1: DH Matrix ───────────────────────────────────
def DH(a, alpha_deg, d, theta_rad):
    """One link's 4x4 transformation matrix"""
    alpha = np.radians(alpha_deg)
    theta = theta_rad            # already radians!
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                 np.cos(alpha),               d],
        [0,              0,                             0,                            1]
    ])

# ── STEP 2: Forward Kinematics ──────────────────────────
def fk_ur5e(theta):
    """
    theta: 6 joint angles in RADIANS
    returns: 4x4 transformation matrix
    last column = (x, y, z) of end effector
    """
    T = DH(0,       90,  0.1625, theta[0])
    T = T @ DH(-0.425,   0,  0,  theta[1])
    T = T @ DH(-0.3922,  0,  0,  theta[2])
    T = T @ DH(0,        90, 0.1333, theta[3])
    T = T @ DH(0,       -90, 0.0997, theta[4])
    T = T @ DH(0,         0, 0.0996, theta[5])
    return T

# ── STEP 3: Jacobian ────────────────────────────────────
def jacobian_ur5e(theta):
    """
    Builds 6x6 Jacobian matrix
    Each column = how that joint affects end effector
    """
    transforms = [
        DH(0,       90,  0.1625, theta[0]),
        DH(-0.425,   0,  0,      theta[1]),
        DH(-0.3922,  0,  0,      theta[2]),
        DH(0,        90, 0.1333, theta[3]),
        DH(0,       -90, 0.0997, theta[4]),
        DH(0,         0, 0.0996, theta[5]),
    ]

    # build chain: [origin, T1, T12, T123, T1234, T12345, T_total]
    T = [np.eye(4)]
    for t in transforms:
        T.append(T[-1] @ t)

    ee = T[6][:3, 3]   # end effector position

    J = np.zeros((6, 6))
    for i in range(6):
        z = T[i][:3, 2]              # z-axis of joint i
        o = T[i][:3, 3]              # position of joint i
        J[:3, i] = np.cross(z, ee - o)  # linear velocity
        J[3:, i] = z                     # angular velocity
    return J

# ── Helper: Clamp joints ────────────────────────────────
def clamp_joints(theta):
    for i in range(6):
        lo, hi = JOINT_LIMITS[i]
        theta[i] = np.clip(theta[i], lo, hi)
    return theta

# ── Helper: Get all joint positions ─────────────────────
def get_joint_positions(theta):
    transforms = [
        DH(0,       90,  0.1625, theta[0]),
        DH(-0.425,   0,  0,      theta[1]),
        DH(-0.3922,  0,  0,      theta[2]),
        DH(0,        90, 0.1333, theta[3]),
        DH(0,       -90, 0.0997, theta[4]),
        DH(0,         0, 0.0996, theta[5]),
    ]
    T = [np.eye(4)]
    for t in transforms:
        T.append(T[-1] @ t)
    return np.array([Ti[:3, 3] for Ti in T])

# ── STEP 4: Inverse Kinematics ──────────────────────────
def ik_ur5e(target, theta_init, max_iter=1000, eps=1e-3, step=0.5):
    """
    Given target (x,y,z), find joint angles
    theta_init: starting guess in RADIANS
    returns: joint angles in RADIANS
    """
    theta = np.array(theta_init, dtype=float)
    error = np.zeros(3)

    print(f"\n{'─'*50}")
    print(f"  TARGET: {np.round(target, 3)}")

    # STEP 4a: FK — where is hand NOW?
    T_now = fk_ur5e(theta)
    current = T_now[:3, 3]
    print(f"  FK (current pos): {np.round(current, 3)}")
    print(f"  Initial error:    {np.round(np.linalg.norm(target - current), 4)} m")
    print(f"  Running IK...")

    # STEP 4b: Newton-Raphson loop
    for i in range(max_iter):
        T       = fk_ur5e(theta)
        current = T[:3, 3]
        error   = target - current

        if np.linalg.norm(error) < eps:
            print(f"  ✓ Converged in {i} iterations | error = {np.linalg.norm(error):.5f} m")
            return theta

        J     = jacobian_ur5e(theta)
        J_pos = J[:3, :]

        # damped least squares (avoids singularities)
        lam   = 0.01
        J_dls = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lam**2 * np.eye(3))

        delta  = J_dls @ error
        theta += step * delta
        theta  = clamp_joints(theta)

    print(f"  ✗ Did not converge | error = {np.linalg.norm(error):.5f} m")
    return theta

# ── STEP 5: Animate movement ────────────────────────────
def animate(theta_start, theta_end, target, label="Moving", steps=60):
    """Smoothly animate robot from theta_start to theta_end"""
    plt.ion()
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')

    for s in range(steps + 1):
        t     = s / steps                               # 0.0 → 1.0
        theta = theta_start + t * (theta_end - theta_start)  # interpolate
        pts   = get_joint_positions(theta)

        ax.cla()

        # draw robot links
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                'o-', color='royalblue', linewidth=3, markersize=7)

        # highlight end effector
        ax.scatter(*pts[-1], color='red', s=120, zorder=5, label='End Effector')

        # show target
        ax.scatter(*target, color='lime', s=150,
                   marker='*', zorder=5, label='Target')

        # draw line from hand to target
        ax.plot([pts[-1, 0], target[0]],
                [pts[-1, 1], target[1]],
                [pts[-1, 2], target[2]],
                'r--', linewidth=1, alpha=0.5)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0,  1.2])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'UR5e — {label}  |  step {s}/{steps}')
        ax.legend(loc='upper right')

        plt.pause(0.03)

    plt.ioff()

# ═══════════════════════════════════════════════════════
#  MAIN — Home → FK → IK → Animate → Repeat
# ═══════════════════════════════════════════════════════

# Home position (standard UR5e home)
THETA_HOME = np.radians([0, -90, 0, -90, 0, 0])

# Pick and place targets
PICK_TARGET  = np.array([0.5,  0.2,  0.3])   # pick object here
PLACE_TARGET = np.array([0.5, -0.2,  0.3])   # place object here

print("═" * 50)
print("  UR5e Pick and Place — Built from Scratch!")
print("═" * 50)

# ── Move 1: Home → Pick ─────────────────────────────────
print("\n[1] HOME → PICK")
theta_pick = ik_ur5e(PICK_TARGET, THETA_HOME)
print(f"    Joint angles (deg): {np.round(np.degrees(theta_pick), 1)}")
animate(THETA_HOME, theta_pick, PICK_TARGET, label="Home → Pick")

# ── Move 2: Pick → Place ────────────────────────────────
print("\n[2] PICK → PLACE")
theta_place = ik_ur5e(PLACE_TARGET, theta_pick)
print(f"    Joint angles (deg): {np.round(np.degrees(theta_place), 1)}")
animate(theta_pick, theta_place, PLACE_TARGET, label="Pick → Place")

# ── Move 3: Place → Home ────────────────────────────────
theta_home = ik_ur5e(np.array([0, 0, 0.5]), theta_place)
print("\n[3] PLACE → HOME")
animate(theta_place, theta_home, np.array([0, 0, 0.5]), label="Place → Home")

print("\n" + "═"*50)
print("  Pick and Place Complete! 🎉")
print("═"*50)

plt.show()