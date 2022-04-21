import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Define geometry
semi_span = 5
num_elements = 2
L_e = 2*semi_span/num_elements
r = 1     # Distance from C.G.

# Define structural properties
I_ext = 10      # External mass
I_b = 100        # Moment of inertia per unit length
mu = 100        # Beam mass per unit length
GJ = 1e2        # Bending stiffness
I_total = I_ext + 2*semi_span*r**2 + 2*I_b*semi_span

# Simulation parameters
tstart = 0
tstop = 10
dt = 0.01
alpha = 0.0001      # Numerical damping for Newmark-beta
t = np.arange(tstart, tstop+1, dt)
Ft = 10*np.ones_like(t)
F_ext = 10

# Define Element Matrices
K_e = np.array([[GJ/L_e, -GJ/L_e],
                [-GJ/L_e, GJ/L_e]])

M_e = mu*L_e/420*np.array([[140*I_b/mu, 70*I_b/mu],
                           [70*I_b/mu, 140*I_b/mu]])

R_e = np.array([[I_b*L_e/2, I_b*L_e/2]])

# Assemble Global Matrices
K_global = np.zeros(((num_elements+1)+1, (num_elements+1)+1))
M_global = np.zeros(((num_elements+1)+1, (num_elements+1)+1))
R_global = np.zeros((1, (num_elements+1)+1))

for element in range(num_elements):
    r, c = element, element
    K_global[r:r+K_e.shape[0], c:c+K_e.shape[1]] += K_e
    M_global[r:r+M_e.shape[0], c:c+M_e.shape[1]] += M_e
    R_global[0:1, c:c+R_e.shape[1]] += R_e

R_globalT = np.transpose(R_global)
M_global[-1:, 0:R_global.shape[1]] += R_global
M_global[0:R_globalT.shape[0], -1:] += R_globalT
M_global[-1,-1] = I_total

# Clamped-Free BC
K_global = np.delete(K_global, int(num_elements/2), 0)
K_global = np.delete(K_global, int(num_elements/2), 1)

M_global = np.delete(M_global, int(num_elements/2), 0)
M_global = np.delete(M_global, int(num_elements/2), 1)

# Additional matrix for Newmark Beta Method
zero_matrix = np.zeros((num_elements+1, num_elements+1))
identity = np.eye(num_elements+1)

# Newmark-beta Method (Avg Acceleration)
phi = 0.5 + alpha
beta = 0.25*(phi+0.5)**2

S_1 = M_global + dt**2*beta*K_global
S_2 = dt**2*(0.5-beta)*K_global

A_1 = np.concatenate((zero_matrix, zero_matrix, S_1), axis=1)
A_2 = np.concatenate((zero_matrix, -identity, dt*phi*identity), axis=1)
A_3 = np.concatenate((-identity, zero_matrix, dt**2*beta*identity), axis=1)

A = np.concatenate((A_1, A_2, A_3), axis = 0)

B_1 = np.concatenate((K_global, K_global*dt, S_2), axis=1)
B_2 = np.concatenate((zero_matrix, identity, (1-phi)*dt*identity), axis=1)
B_3 = np.concatenate((identity, dt*identity, (0.5-beta)*dt**2*identity), axis=1)

B = np.concatenate((B_1, B_2, B_3), axis = 0)

# Set initial conditions
q = np.zeros(num_elements+1)
q_dot = np.zeros(num_elements+1)
q_dotdot = np.zeros(num_elements+1)
Q_old = np.concatenate((q, q_dot, q_dotdot), axis=0)

F_old = np.zeros(num_elements+1)
F_old = np.concatenate((F_old, np.zeros(num_elements+1), np.zeros(num_elements+1)), axis=0)

F_old[0] = 0
F_old[num_elements-1] = 0
F_old[num_elements] = F_old[0] + F_old[num_elements-1] + F_ext

CG = []
tip_left = []
tip_right = []

tip_left_local = []
tip_right_local = []

for time in t:
    RHS = F_old - np.matmul(B, Q_old)
    Q_new = np.matmul(inv(A), RHS)
    CG.append(Q_new[num_elements])
    tip_left.append(Q_new[0] + Q_new[num_elements])
    tip_right.append(Q_new[num_elements-1] + Q_new[num_elements])

    tip_left_local.append(Q_new[0])
    tip_right_local.append(Q_new[num_elements-1])

    Q_old = Q_new

    plt.figure(1)
plt.plot(t, CG)
plt.plot(t, tip_left)
plt.plot(t, tip_right)
plt.xlabel("time (s)")
plt.ylabel("angle")
plt.legend(("CG", "Left Tip", "Right Tip"))

plt.figure(2)
plt.plot(t, tip_left_local)
plt.plot(t, tip_right_local)
plt.xlabel("time (s)")
plt.ylabel("angle")
plt.legend(("Left Tip", "Right Tip"))

plt.show()