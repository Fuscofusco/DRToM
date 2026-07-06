import numpy as np

# Your function (unchanged)
def phase3(N, M):
    q = np.zeros((N, 4))
    p = np.zeros((N, 4))
    for i in range(N):
        phi = np.random.uniform(0, 2 * np.pi)
        costheta = np.random.uniform(-1, 1)
        sintheta = np.sqrt(1 - costheta**2)
        q[i, 0] = -np.log(np.random.uniform(0, 1) * np.random.uniform(0, 1))
        q[i, 1] = q[i, 0] * np.cos(phi) * sintheta
        q[i, 2] = q[i, 0] * np.sin(phi) * sintheta
        q[i, 3] = q[i, 0] * costheta
    qsum = q.sum(axis=0)
    mass = np.sqrt(qsum[0]**2 - np.dot(qsum[1:], qsum[1:]))
    gamma = qsum[0] / mass
    beta = -qsum[1:] / mass
    for i in range(N):
        bq = np.dot(beta, q[i, 1:])
        p[i, 0] = (M / mass) * (gamma * q[i, 0] + bq)
        p[i, 1:] = (M / mass) * (q[i, 1:] + beta * (q[i, 0] + bq / (1 + gamma)))
    return p

# =========================
# Test masslessness
# =========================
N_events = 100000
N_particles = 2   # or whatever you want
M = 1000.0        # total invariant mass

max_deviation = 0.0

for _ in range(N_events):
    p = phase3(N_particles, M)
    
    # Compute m^2 = E^2 - |p|^2 for each particle
    m2 = p[:, 0]**2 - np.sum(p[:, 1:]**2, axis=1)
    
    max_deviation = max(max_deviation, np.max(np.abs(m2)))

print("Max |m^2| deviation from 0:", max_deviation)