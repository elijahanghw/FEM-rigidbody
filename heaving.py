import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.linalg import inv
from scipy.linalg import eig

# Define geometry
semi_span = 5
num_elements = 4
L_e = 2*semi_span/num_elements

# Define structural properties
M_ext = 0     # External mass
mu = 20      # Beam mass per unit length
EI = 1e3     # Bending stiffness
M_total = M_ext + 2*semi_span*mu

# Simulation parameters
tstart = 0
tstop = 10
TIME_STEPS = 1000
dt = 0.01
alpha = 0.0001
t = np.arange(tstart, tstop+1, dt)
Ft = 1*np.ones_like(t)

# Define Element Matrices
K_e = np.array([[12*EI/(L_e**3), 6*EI/(L_e**2), -12*EI/(L_e**3), 6*EI/(L_e**2)],
                [6*EI/(L_e**2), 4*EI/L_e, -6*EI/(L_e**2), 2*EI/L_e],
                [-12*EI/(L_e**3), -6*EI/(L_e**2), 12*EI/(L_e**3), -6*EI/(L_e**2)],
                [6*EI/(L_e**2), 2*EI/L_e, -6*EI/(L_e**2), 4*EI/L_e]])

M_e = mu*L_e/420*np.array([[156, 22*L_e, 54, -13*L_e],
                           [22*L_e, 4*L_e**2, 13*L_e, -3*L_e**2],
                           [54, 13*L_e, 156, -22*L_e],
                           [-13*L_e, -3*L_e**2, -22*L_e, 4*L_e**2]])

# R_e = np.array([[mu*L_e/2, mu*L_e**2/12, 5*mu*L_e/6, -mu*L_e**2/12]])
R_e = np.array([[0, 0, 0, 0]])

# M_e = np.array([[1, 1, 1, 1],
#                 [1, 1, 1, 1],
#                 [1, 1, 1, 1],
#                 [1, 1, 1, 1]])

# K_e = np.array([[1, 1, 1, 1],
#                 [1, 1, 1, 1],
#                 [1, 1, 1, 1],
#                 [1, 1, 1, 1]])

# R_e = np.array([[1, 1, 1, 1]])

# Assemble Global Matrices
K_global = np.zeros((2*(num_elements+1)+1, 2*(num_elements+1)+1))
M_global = np.zeros((2*(num_elements+1)+1, 2*(num_elements+1)+1))
R_global = np.zeros((1, 2*(num_elements+1)+1))

for element in range(num_elements):
    r, c = element*2, element*2
    K_global[r:r+K_e.shape[0], c:c+K_e.shape[1]] += K_e
    M_global[r:r+M_e.shape[0], c:c+M_e.shape[1]] += M_e
    R_global[0:1, c:c+R_e.shape[1]] += R_e

R_globalT = np.transpose(R_global)
M_global[-1:, 0:R_global.shape[1]] += R_global
M_global[0:R_globalT.shape[0], -1:] += R_globalT
M_global[-1,-1] = M_total

# Clamped-Free BC
K_global = np.delete(K_global, num_elements, 0)
K_global = np.delete(K_global, num_elements, 0)
K_global = np.delete(K_global, num_elements, 1)
K_global = np.delete(K_global, num_elements, 1)

M_global = np.delete(M_global, num_elements, 0)
M_global = np.delete(M_global, num_elements, 0)
M_global = np.delete(M_global, num_elements, 1)
M_global = np.delete(M_global, num_elements, 1)

# Additional matrix for Newmark Beta Method
zero_matrix = np.zeros((2*num_elements+1, 2*num_elements+1))
identity = np.eye(2*num_elements+1)

# Newmark Beta Method (Avg Acceleration)
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
q = np.zeros(2*num_elements+1)
q_dot = np.zeros(2*num_elements+1)
q_dotdot = np.zeros(2*num_elements+1)
Q_old = np.concatenate((q, q_dot, q_dotdot), axis=0)

F_old = np.zeros(2*num_elements+1)
F_old = np.concatenate((F_old, np.zeros(2*num_elements+1), np.zeros(2*num_elements+1)), axis=0)

F_old[0] = 1
F_old[2*num_elements-2] = -1
F_old[2*num_elements] = F_old[0] + F_old[2*num_elements-2]

CG = []
tip_left = []
tip_right = []
time = []

for T in range(TIME_STEPS):
    print(T)
    RHS = F_old - np.matmul(B, Q_old)
    Q_new = np.matmul(inv(A), RHS)

    CG.append(Q_new[2*num_elements])
    tip_left.append(Q_new[0] + Q_new[2*num_elements])
    tip_right.append(Q_new[2*num_elements-2] + Q_new[2*num_elements])
    time.append(T*dt)

    Q_old = Q_new
    #F_old[3*(num_elements-1)] = 0

plt.plot(time, CG)
plt.plot(time, tip_left)
plt.plot(time, tip_right)
plt.xlabel("time (s)")
plt.ylabel("displacement (m)")
plt.legend(("CG", "Left Tip", "Right Tip"))
plt.show()