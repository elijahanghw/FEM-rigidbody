import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Define structural properties
M_ext = 20     # External mass
I_ext = 10      # External moment of inertia
r = 0.1           # Distance from C.G. to beam mass axis
a = -0.1           # Distance of elastic axis from mass axis
M_b = 10        # Body mass
I_b = 10       # Body moment of inertia
K = 20          # Spring Stiffness
K_T = 20         # Bending stiffness

# Simulation parameters
tstart = 0
tstop = 10
dt = 0.01
alpha = 0.0001      # Numerical damping for Newmark-beta
t = np.arange(tstart, tstop+1, dt)
Force_ext = 10
Moment_ext = 0

# Define matrices
K = np.array([[K, 0, 0, 0,],
             [0, K_T, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]])

M = np.array([[M_b, M_b*a, M_b, -M_b*r],
              [M_b*a, I_b+M_b*a**2, M_b*a, I_b-M_b*a*r],
              [M_b, M_b*a, M_ext+M_b, 0],
              [M_b*r, I_b-M_b*a*r, 0, I_ext+I_b+M_b*r**2]])
    


# Additional matrix for Newmark Beta Method
zero_matrix = np.zeros((4, 4))
identity = np.eye(4)

# Newmark-beta Method (Avg Acceleration)
phi = 0.5 + alpha
beta = 0.25*(phi+0.5)**2

S_1 = M + dt**2*beta*K
S_2 = dt**2*(0.5-beta)*K

A_1 = np.concatenate((zero_matrix, zero_matrix, S_1), axis=1)
A_2 = np.concatenate((zero_matrix, -identity, dt*phi*identity), axis=1)
A_3 = np.concatenate((-identity, zero_matrix, dt**2*beta*identity), axis=1)

A = np.concatenate((A_1, A_2, A_3), axis = 0)

B_1 = np.concatenate((K, K*dt, S_2), axis=1)
B_2 = np.concatenate((zero_matrix, identity, (1-phi)*dt*identity), axis=1)
B_3 = np.concatenate((identity, dt*identity, (0.5-beta)*dt**2*identity), axis=1)

B = np.concatenate((B_1, B_2, B_3), axis = 0)

# Set initial conditions
q = np.zeros(4)
q_dot = np.zeros(4)
q_dotdot = np.zeros(4)
Q_old = np.concatenate((q, q_dot, q_dotdot), axis=0)

F_old = np.zeros(4)
F_old = np.concatenate((F_old, np.zeros(4), np.zeros(4)), axis=0)

F_old[0] = 0
F_old[1] = 0
F_old[2] = F_old[0] + Force_ext
F_old[3] = -(r+a)*F_old[0] + F_old[1] + Moment_ext

CG_heave = []
CG_twist = []

Mass_heave = []
Mass_pitch = []

Mass_heave_local = []
Mass_pitch_local = []

for time in t:
    RHS = F_old - np.matmul(B, Q_old)
    Q_new = np.matmul(inv(A), RHS)
    CG_heave.append(Q_new[2])
    CG_twist.append(Q_new[3])
    Mass_heave.append(Q_new[0] + Q_new[2] - r*Q_new[3])
    Mass_pitch.append(Q_new[1] + Q_new[3])
    Mass_heave_local.append(Q_new[0])
    Mass_pitch_local.append(Q_new[1])

    Q_old = Q_new

plt.figure(1)
plt.plot(t, CG_heave)
plt.plot(t, Mass_heave)
plt.xlabel("time (s)")
plt.ylabel("displacement")
plt.legend(("CG", "Mass"))

plt.figure(2)
plt.plot(t, CG_twist)
plt.plot(t, Mass_pitch)
plt.xlabel("time (s)")
plt.ylabel("angle")
plt.legend(("CG", "Mass"))

plt.show()