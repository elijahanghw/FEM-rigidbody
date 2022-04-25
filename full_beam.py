import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Define geometry
semi_span = 6.096
num_elements = 8
L_e = 2*semi_span/num_elements

# Define structural properties
M_ext = 10     # External mass
I_ext = 100      # External moment of inertia
r = 0.1           # Distance from C.G. to beam mass axis
a = -0.18288  # Distance of elastic axis from mass axis
mu = 10     # Beam mass per unit length
I_alpha = 8.64       # Moment of inertia per unit length
EI = 9.77e2    # Bending stiffness
GJ = 0.99e2         # Torsional stiffness

M_total = M_ext + 2*semi_span*mu
I_total = 2*I_alpha*semi_span + I_ext + 2*semi_span*mu*r**2

# Simulation parameters
tstart = 0
tstop = 10
dt = 0.01
alpha = 0.0001      # Numerical damping for Newmark-beta
t = np.arange(tstart, tstop+1, dt)

# Define Element Matrices
K_e = np.array([[12*EI/(L_e**3), 6*EI/(L_e**2), 0, -12*EI/(L_e**3), 6*EI/(L_e**2), 0],
                    [6*EI/(L_e**2), 4*EI/L_e, 0, -6*EI/(L_e**2), 2*EI/L_e, 0],
                    [0, 0, GJ/L_e, 0, 0, -GJ/L_e],
                    [-12*EI/(L_e**3), -6*EI/(L_e**2), 0, 12*EI/(L_e**3), -6*EI/(L_e**2), 0],
                    [6*EI/(L_e**2), 2*EI/L_e, 0, -6*EI/(L_e**2), 4*EI/L_e, 0],
                    [0, 0, -GJ/L_e, 0, 0, GJ/L_e]])

M_e = mu*L_e/420*np.array([[156, 22*L_e, 0, 54, -13*L_e, 0],
                            [22*L_e, 4*L_e**2, 0, 13*L_e, -3*L_e**2, 0],
                            [0, 0, 140*I_alpha/mu, 0, 0, 70*I_alpha/mu],
                            [54, 13*L_e, 0, 156, -22*L_e, 0],
                            [-13*L_e, -3*L_e**2, 0, -22*L_e, 4*L_e**2, 0],
                            [0, 0, 70*I_alpha/mu, 0, 0, 140*I_alpha/mu]])


S_e = mu*L_e*a/60*np.array([[0, 0, 21, 0, 0, 9],
                                [0, 0, 3*L_e, 0, 0, 2*L_e],
                                [21, 3*L_e, 0, 9, -2*L_e, 0],
                                [0, 0, 9, 0, 0, 21],
                                [0, 0, -2*L_e, 0, 0, -3*L_e],
                                [9, 2*L_e, 0, 21, -3*L_e, 0]])

M_e = M_e + S_e

Rh_e = np.array([[mu*L_e/2, mu*L_e**2/12, mu*a*L_e/2, mu*L_e/2, -mu*L_e**2/12, mu*a*L_e/2]])
Rt_e = np.array([[-mu*L_e*r/2, -mu*L_e**2*r/12, (I_alpha-mu*a*r)*L_e/2, -mu*L_e*r/2, mu*L_e**2*(a-r)/12, (I_alpha-mu*a*r)*L_e/2]])

# Assemble Global Matrices
K_global = np.zeros((3*(num_elements+1)+2, 3*(num_elements+1)+2))
M_global = np.zeros((3*(num_elements+1)+2, 3*(num_elements+1)+2))
Rh_global = np.zeros((1, 3*(num_elements+1)+2))
Rt_global = np.zeros((1, 3*(num_elements+1)+2))

for element in range(num_elements):
    row, col = 3*element, 3*element
    K_global[row:row+K_e.shape[0], col:col+K_e.shape[1]] += K_e
    M_global[row:row+M_e.shape[0], col:col+M_e.shape[1]] += M_e
    Rh_global[0:1, col:col+Rh_e.shape[1]] += Rh_e
    Rt_global[0:1, col:col+Rt_e.shape[1]] += Rt_e

Rh_globalT = np.transpose(Rh_global)
Rt_globalT = np.transpose(Rt_global)

M_global[-2:-1, 0:Rh_global.shape[1]] += Rh_global
M_global[0:Rh_globalT.shape[0], -2:-1] += Rh_globalT

M_global[-1:, 0:Rt_global.shape[1]] += Rt_global
M_global[0:Rt_globalT.shape[0], -1:] += Rt_globalT

M_global[-2,-2] = M_total
M_global[-1,-1] = I_total

# Clamped-Free BC
K_global = np.delete(K_global, int(3/2*num_elements), 0)
K_global = np.delete(K_global, int(3/2*num_elements), 0)
K_global = np.delete(K_global, int(3/2*num_elements), 0)
K_global = np.delete(K_global, int(3/2*num_elements), 1)
K_global = np.delete(K_global, int(3/2*num_elements), 1)
K_global = np.delete(K_global, int(3/2*num_elements), 1)

M_global = np.delete(M_global, int(3/2*num_elements), 0)
M_global = np.delete(M_global, int(3/2*num_elements), 0)
M_global = np.delete(M_global, int(3/2*num_elements), 0)
M_global = np.delete(M_global, int(3/2*num_elements), 1)
M_global = np.delete(M_global, int(3/2*num_elements), 1)
M_global = np.delete(M_global, int(3/2*num_elements), 1)

# Additional matrix for Newmark Beta Method
zero_matrix = np.zeros((3*num_elements+2, 3*num_elements+2))
identity = np.eye(3*num_elements+2)

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
q = np.zeros(3*num_elements+2)
q_dot = np.zeros(3*num_elements+2)
q_dotdot = np.zeros(3*num_elements+2)
Q_old = np.concatenate((q, q_dot, q_dotdot), axis=0)

F_old = np.zeros(3*num_elements+2)
F_old = np.concatenate((F_old, np.zeros(3*num_elements+2), np.zeros(3*num_elements+2)), axis=0)

F_old[0] = 0
F_old[3*num_elements-3] = 0
F_old[3*num_elements] = F_old[0] + F_old[num_elements-1] + 10
F_old[3*num_elements+1] = -(r+a)*(F_old[0] + F_old[num_elements-1])

CG_heave = []
CG_twist = []
tip_left_heave = []
tip_right_heave = []
tip_left_twist = []
tip_right_twist = []

tip_left_local_heave = []
tip_right_local_heave = []
tip_left_local_twist = []
tip_right_local_twist = []

for time in t:
    RHS = F_old - np.matmul(B, Q_old)
    Q_new = np.matmul(inv(A), RHS)
    CG_heave.append(Q_new[3*num_elements])
    CG_twist.append(Q_new[3*num_elements+1])
    tip_left_heave.append(Q_new[0] + Q_new[3*num_elements] - r*Q_new[3*num_elements+1])
    tip_right_heave.append(Q_new[3*num_elements-3] + Q_new[3*num_elements]- r*Q_new[3*num_elements+1])
    tip_left_twist.append(Q_new[2] + Q_new[3*num_elements+1])
    tip_right_twist.append(Q_new[3*num_elements-1] + Q_new[3*num_elements+1])

    tip_left_local_heave.append(Q_new[0] - r*Q_new[3*num_elements+1])
    tip_right_local_heave.append(Q_new[3*num_elements-3] - r*Q_new[3*num_elements+1])
    tip_left_local_twist.append(Q_new[2])
    tip_right_local_twist.append(Q_new[3*num_elements-1])

    Q_old = Q_new

plt.figure(1)
plt.plot(t, CG_heave)
plt.plot(t, tip_left_heave)
plt.plot(t, tip_right_heave)
plt.xlabel("time (s)")
plt.ylabel("displacement")
plt.legend(("CG", "Left Tip", "Right Tip"))

plt.figure(2)
plt.plot(t, tip_left_local_heave)
plt.plot(t, tip_right_local_heave)
plt.xlabel("time (s)")
plt.ylabel("displacement")
plt.legend(("Left Tip", "Right Tip"))

plt.figure(3)
plt.plot(t, CG_twist)
plt.plot(t, tip_left_twist)
plt.plot(t, tip_right_twist)
plt.xlabel("time (s)")
plt.ylabel("angle")
plt.legend(("CG", "Left Tip", "Right Tip"))

plt.figure(4)
plt.plot(t, tip_left_local_twist)
plt.plot(t, tip_right_local_twist)
plt.xlabel("time (s)")
plt.ylabel("angle")
plt.legend(("Left Tip", "Right Tip"))

plt.show()