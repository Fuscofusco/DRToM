import os
import numpy as np
import lhapdf
import random
import re
from collections import Counter
from collections import defaultdict
import math 


# Global debug set
debug = False 

#============================================================
#=================== GENERATION PARAMETERS ==================
#============================================================

# Load PDF set
PDFSet = "cteq6l1"
PDF = lhapdf.mkPDF(PDFSet, 0)

# Number of integration slices
MC = 5



#============================================================
#====================== BASELINE FUNCTIONS ==================
#============================================================

def Integrate(Func,args,N,a,b,x):
    # Double integral
    value = 0
    for i in range(1,N+1):
        Y = a+((i-(1/2))*((b-a)/N))
        c = -(x - np.fabs(Y))
        d = x - np.fabs(Y)
        for j in range(1,N+1):
            y = c+((j-(1/2))*((d-c)/N))
            params = [Y,y]
            value += ((b-a)/N)*((d-c)/N)*Func(params,*args)
    return value


def boost(gamma, beta, p):
    # Lorentz boost
    b2 = np.dot(beta, beta)
    b = np.sqrt(b2)
    b_hat = beta / b if b > 0 else beta
    Ep = p[0]
    pp = p[1:]
    E = gamma * (Ep + np.dot(pp, beta))
    P = pp + (gamma-1.0) * np.dot(pp, b_hat) * b_hat + gamma*Ep*beta

    m2 = E*E - np.dot(P,P)      # Invariant mass squared after boost

    if m2 < -9e-5:              
        raise ValueError(f"Boost produced spacelike 4-vector: m^2={m2}")

    if m2 < 0:                  # Tiny negative (round-off)
        E = np.sqrt(np.dot(P,P)) 

    return np.concatenate(([E], P))


#============================================================================
#========================= OUTGOING KINEMATICS ==============================
#============================================================================

def phase2(N, M, debug=False):
    # 2D phase space generator with N massless momenta
    # 1) Generate N massless momenta in 2D
    q = np.zeros((N, 4))
    for i in range(N):
        phi = np.random.uniform(0, 2*np.pi)
        energy = -np.log(np.random.uniform(0, 1) * np.random.uniform(0, 1))
        q[i, 0] = energy
        q[i, 1] = energy * np.cos(phi)
        q[i, 2] = energy * np.sin(phi)
        q[i, 3] = 0.0

    # 2) Boost to rest frame of total momentum and scale to mass M
    qsum = q.sum(axis=0)
    mass = np.sqrt(qsum[0]**2 - np.dot(qsum[1:], qsum[1:]))
    gamma = qsum[0] / mass
    beta = -qsum[1:] / mass

    p = np.zeros_like(q)
    for i in range(N):
        bq = np.dot(beta, q[i, 1:])
        p[i, 0] = (M/mass) * (gamma*q[i, 0] + bq)
        p[i, 1:] = (M/mass) * (q[i, 1:] + beta*(q[i, 0] + bq/(1+gamma)))

    # 3) Apply a single random 3D rotation
    def random_rotation_matrix():
        # Generate a random 3D rotation matrix using the uniform quaternion method.
        u1, u2, u3 = np.random.uniform(0,1,3)
        q1 = np.sqrt(1-u1) * np.sin(2*np.pi*u2)
        q2 = np.sqrt(1-u1) * np.cos(2*np.pi*u2)
        q3 = np.sqrt(u1) * np.sin(2*np.pi*u3)
        q4 = np.sqrt(u1) * np.cos(2*np.pi*u3)
        R = np.array([
            [1-2*(q2**2 + q3**2),   2*(q1*q2 - q3*q4),     2*(q1*q3 + q2*q4)],
            [2*(q1*q2 + q3*q4),     1-2*(q1**2 + q3**2),   2*(q2*q3 - q1*q4)],
            [2*(q1*q3 - q2*q4),     2*(q2*q3 + q1*q4),     1-2*(q1**2 + q2**2)]
        ])
        return R

    R = random_rotation_matrix()
    p[:, 1:] = np.dot(p[:, 1:], R.T)  # Rotate all spatial components at once

    # Optional debugging
    if debug:
        eps = 1e-12
        total_p = p.sum(axis=0)
        if abs(M - total_p[0]) > eps:
            print("Energy not conserved:", M, total_p[0])
        if np.linalg.norm(total_p[1:]) > eps:
            print("Momentum not conserved:", total_p[1:])

    return p


def phase3(N, M):
    # 3D phase space generator with N massless momenta,
    # Same as 'phase2' but we don't force the z-component to be 0 and dont apply a rotation
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


def QCD_2to2_outgoingkin(interaction_name, M, RandomY, Randomy, outIDs, QuarkMasses):
    # Outgoing 2->2 parton kinematics in CoM frame
    m1 = get_quark_mass(abs(outIDs[0]))
    m2 = get_quark_mass(abs(outIDs[1]))
    
    def pT_general(M, m1, m2, y):
        # pT in CoM from rapidity 
        a = (M**2 - m1**2 - m2**2) / 2
        A = np.cosh(2*y)**2 - 1
        B = m1**2 * np.cosh(2*y)**2 + m2**2 * np.cosh(2*y)**2 + 2*a
        C = m1**2 * m2**2 - a**2
        num = -B + np.sqrt(B**2 - 4*A*C)
        dem = 2*A
        pT = np.sqrt( num / dem )
        return pT

    y1, y2 = Randomy, -Randomy
    phi = random.uniform(0, 2*math.pi)

    # Case 1: massless–massless
    if m1 == 0.0 and m2 == 0.0:
        #pt = 0.5 * M / math.cosh(Randomy)
        pt = pT_general(M, m1, m2, Randomy)
        E1 = pt * math.cosh(y1)
        E2 = pt * math.cosh(y2)
        px1, py1 = pt * math.cos(phi), pt * math.sin(phi)
        px2, py2 = -px1, -py1
        pz1 = E1 * math.tanh(y1)
        pz2 = E2 * math.tanh(y2)
        
    # Case 2: equal mass massive particles
    elif m1 == m2 and m1 > 0.0:
        m = m1
        #pt = 0.5 / math.cosh(Randomy) * math.sqrt(M**2 - 4*m**2 * math.cosh(Randomy)**2)
        pt = pT_general(M, m1, m2, Randomy)
        E1 = math.sqrt(m**2 + pt**2) * math.cosh(y1)
        E2 = math.sqrt(m**2 + pt**2) * math.cosh(y2)
        px1, py1 = pt * math.cos(phi), pt * math.sin(phi)
        px2, py2 = -px1, -py1
        pz1 = E1 * math.tanh(y1)
        pz2 = E2 * math.tanh(y2)

    # Case 3: massless + massive
    elif (m1 == 0 and m2 > 0) or (m2 == 0 and m1 > 0):
        # Identify which particle is massive
        if m1 == 0:
            idx_massless, idx_massive = 0, 1
            Qmass = m2
        else:
            idx_massless, idx_massive = 1, 0
            Qmass = m1

        pt = pT_general(M, m1, m2, Randomy)
        E_massless = pt * math.cosh(y1)
        E_massive  = math.sqrt(Qmass**2 + pt**2) * math.cosh(y2)
        px = pt * math.cos(phi)
        py = pt * math.sin(phi)
        pz_massless = E_massless * math.tanh(y1)
        pz_massive  = E_massive * math.tanh(y2)

        # Assign to outgoing 4-momenta in correct order
        if idx_massless == 0:
            E1, px1, py1, pz1 = E_massless,  px,  py,  pz_massless
            E2, px2, py2, pz2 = E_massive,  -px, -py,  pz_massive
        else:
            E1, px1, py1, pz1 = E_massive,  -px, -py,  pz_massive
            E2, px2, py2, pz2 = E_massless,  px,  py,  pz_massless


    # Case 4: both massive, different masses
    elif m1 > 0 and m2 > 0 and m1 != m2:
        pt = pT_general(M, m1, m2, Randomy)
        E1 = math.sqrt(m1**2 + pt**2) * math.cosh(y1)
        E2 = math.sqrt(m2**2 + pt**2) * math.cosh(y2)
        px1, py1 = pt * math.cos(phi), pt * math.sin(phi)
        px2, py2 = -px1, -py1
        pz1 = E1 * math.tanh(y1)
        pz2 = E2 * math.tanh(y2)

    else:
        raise ValueError("Case not implemented (exotic combination)")

    outgoing_particles_CoM = np.array([
        [E1, px1, py1, pz1],
        [E2, px2, py2, pz2]
    ])

    return outgoing_particles_CoM, pt


def check_event_physical(particles, tol=9e-5, label=""):
    # If we generate an unphysical event it needs to be kicked out 
    # tol needs to be around 1e-5 for 2 -> 2 generation (see test.py in Analysis area) 
    total_p = np.zeros(4)
    
    for i, p in enumerate(particles):
        E = p[0]
        pvec = p[1:]
        m2 = E*E - np.dot(pvec, pvec)

        if m2 < -tol:
            print(f"[FAIL] {label} particle {i} spacelike: m^2 = {m2}")
            return False

        total_p += p

    return True

#=================================================================================
#======================= MATRIX ELEMENTS AND COLOUR FLOW =========================
#=================================================================================

# If not symmetric in u and t then another term with the flip is needed
# If final state has the same particle then divide by 2

# (1) gg → gg, already symmetric in u/t
def M2_gg_gg(shat, that, uhat):
    fin_state = 1/2
    return fin_state * ( (9 / 2) * (3 - (that * uhat) / shat**2 - (shat * uhat) / that**2 - (shat * that) / uhat**2) )

# (2) gg → qq̄, already symmetric in u/t
def M2_gg_qqx(shat, that, uhat):
    return (1/6) * (that**2 + uhat**2) / (that * uhat) - (3/8) * (that**2 + uhat**2) / shat**2

# (3) qq̄ → gg, already symmetric in u/t
def M2_qqx_gg(shat, that, uhat):
    fin_state = 1/2
    return fin_state * ( (32 / 27) * (that**2 + uhat**2) / (that * uhat) - (8 / 3) * (that**2 + uhat**2) / shat**2 ) 

# (4) gq → gq
def M2_gq_gq(shat, that, uhat):
    term1 = -4/9 * (shat**2 + uhat**2) / (shat * uhat) + (uhat**2 + shat**2) / that**2
    term2 = -4/9 * (shat**2 + that**2) / (shat * that) + (that**2 + shat**2) / uhat**2
    return term1 + term2

# (5) gq̄ → gq̄
def M2_gqx_gqx(shat, that, uhat):
    term1 = -4/9 * (shat**2 + uhat**2) / (shat * uhat) + (uhat**2 + shat**2) / that**2
    term2 = -4/9 * (shat**2 + that**2) / (shat * that) + (that**2 + shat**2) / uhat**2
    return term1 + term2

# (6) qq' → qq'  
def M2_qqp_qqp(shat, that, uhat):
    term1 = (4 / 9) * (shat**2 + uhat**2) / that**2
    term2 = (4 / 9) * (shat**2 + that**2) / uhat**2
    return term1 + term2

# (7) qq̄′ → qq̄′ 
def M2_qqpx_qqpx(shat, that, uhat):
    term1 = (4 / 9) * (shat**2 + uhat**2) / that**2
    term2 = (4 / 9) * (shat**2 + that**2) / uhat**2
    return term1 + term2

# (8) q̄q̄′ → q̄q̄′  
def M2_qxqpx_qxqpx(shat, that, uhat):
    term1 = (4 / 9) * (shat**2 + uhat**2) / that**2
    term2 = (4 / 9) * (shat**2 + that**2) / uhat**2
    return term1 + term2

# (9) qq̄ → q'q̄', already symmetric in u/t
def M2_qqx_qpqpx(shat, that, uhat):
    return (4 / 9) * (that**2 + uhat**2) / shat**2

# (10) qq → qq, already symmetric in u/t
def M2_qq_qq(shat, that, uhat):
    fin_state = 1/2
    return fin_state * ( (4 / 9) * ((shat**2 + uhat**2) / that**2 + (shat**2 + that**2) / uhat**2) - (8 / 27) * shat**2 / (uhat * that) )

# (11) q̄q̄ → q̄q̄, already symmetric in u/t
def M2_qxqx_qxqx(shat, that, uhat):
    fin_state = 1/2
    return fin_state * ( (4 / 9) * ((shat**2 + uhat**2) / that**2 + (shat**2 + that**2) / uhat**2) - (8 / 27) * shat**2 / (uhat * that) )

# (12) qq̄ → qq̄ 
def M2_qqx_qqx(shat, that, uhat):
    term1 = (4 / 9) * ((shat**2 + uhat**2) / that**2 + (that**2 + uhat**2) / shat**2) - (8 / 27) * uhat**2 / (shat * that)
    term2 = (4 / 9) * ((shat**2 + that**2) / uhat**2 + (uhat**2 + that**2) / shat**2) - (8 / 27) * that**2 / (shat * uhat)
    return term1 + term2


def subprocess_combinations(subprocess):
    # To get possible incoming IDs for each case with the ME
    combinations = []

    if subprocess == "gg_gg":
        func = M2_gg_gg
        combinations.append((21, 21, func))

    elif subprocess == "gg_qqx":
        func = M2_gg_qqx
        combinations.append((21, 21, func))

    elif subprocess == "qqx_gg":
        func = M2_qqx_gg
        for i in range(1, 7):
            combinations.append((i, -i, func))

    elif subprocess == "gq_gq":
        func = M2_gq_gq
        for i in range(1, 7):
            combinations.append((21, i, func))

    elif subprocess == "gqx_gqx":
        func = M2_gqx_gqx
        for i in range(1, 7):
            combinations.append((21, -i, func))

    elif subprocess == "qqp_qqp":
        func = M2_qqp_qqp
        for i in range(1, 7):
            for j in range(i+1, 7):
                combinations.append((i, j, func))

    elif subprocess == "qqpx_qqpx":
        func = M2_qqpx_qqpx
        for i in range(1, 7):
            for j in range(1, 7):
                if i != j:
                    combinations.append((i, -j, func))

    elif subprocess == "qxqpx_qxqpx":
        func = M2_qxqpx_qxqpx
        for i in range(1, 7):
            for j in range(i+1, 7):
                combinations.append((-i, -j, func))

    elif subprocess == "qqx_qpqpx":
        func = M2_qqx_qpqpx
        for i in range(1, 7):
            combinations.append((i, -i, func))

    elif subprocess == "qq_qq":
        func = M2_qq_qq
        for i in range(1, 7):
            combinations.append((i, i, func))

    elif subprocess == "qxqx_qxqx":
        func = M2_qxqx_qxqx
        for i in range(1, 7):
            combinations.append((-i, -i, func))

    elif subprocess == "qqx_qqx":
        func = M2_qqx_qqx
        for i in range(1, 7):
            combinations.append((i, -i, func))

    return combinations


QuarkIDs = [1, 2, 3, 4, 5] #, 6]                             # d, u, s, c, b, t --- Quark PDG IDs
QuarkMasses = [0.0047, 0.0022, 0.096, 1.27, 4.18] #, 173.0]  # approximate MSbar masses (in GeV), we are in massless limit

def get_quark_mass(PDG_ID):
    # Antiquarks have negative PDG IDs, so use abs()
    abs_id = abs(PDG_ID)
    if abs_id in QuarkIDs:
        return QuarkMasses[QuarkIDs.index(abs_id)]
    return 0.0 

        
def PartonOutIDs(MSquaredFunc, ID1, ID2):
    # Outgoing quarks for 2->2
    if MSquaredFunc == M2_gg_gg:                # (1) gg → gg
        return [21, 21]

    elif MSquaredFunc == M2_gg_qqx:             # (2) gg → qq̄
        q = random.choice(QuarkIDs)
        return [q, -q]

    elif MSquaredFunc == M2_qqx_gg:             # (3) qq̄ → gg
        return [21, 21]

    elif MSquaredFunc == M2_gq_gq:              # (4) gq → gq
        return [21, ID2]

    elif MSquaredFunc == M2_gqx_gqx:            # (5) gq̄ → gq̄
        return [21, ID2]

    elif MSquaredFunc == M2_qqp_qqp:            # (6) qq' → qq'
        return [ID1, ID2]

    elif MSquaredFunc == M2_qqpx_qqpx:          # (7) qq̄' → qq̄'
        return [ID1, ID2]

    elif MSquaredFunc == M2_qxqpx_qxqpx:        # (8) q̄q̄' → q̄q̄'
        return [ID1, ID2]

    elif MSquaredFunc == M2_qqx_qpqpx:          # (9) qq̄ → q'q̄'
        qp = random.choice([q for q in QuarkIDs if q != abs(ID1)])
        return [qp, -qp]

    elif MSquaredFunc == M2_qq_qq:              # (10) qq → qq
        return [ID1, ID2]

    elif MSquaredFunc == M2_qxqx_qxqx:          # (11) q̄q̄ → q̄q̄
        return [ID1, ID2]

    elif MSquaredFunc == M2_qqx_qqx:            # (12) qq̄ → qq̄
        return [ID1, ID2]

    else:
        raise ValueError("Unknown matrix element function")


def random_quark_pair():
    q = random.choice(QuarkIDs)
    return [q, -q]

def random_new_quark_pair(exclude):
    others = [x for x in QuarkIDs if x not in exclude]
    qp = random.choice(others)
    return [qp, -qp]

def random_two_distinct_quarks():
    q, qp = random.sample(QuarkIDs, 2)
    return [q, qp]

def random_two_distinct_quark_pairs():
    pair1 = random_quark_pair()
    pair2 = random_quark_pair()
    return pair1 + pair2

def random_qqxqpqpx_ids():
    q = random.choice(QuarkIDs)
    others = [x for x in QuarkIDs if x != q]
    qp = random.choice(others)
    return [q, -q, qp, -qp]



#=========================================================
#======================= ID MAPS =========================
#=========================================================

IDmap_2to2 = {
    # Parton ID maps for 2->2 processes. Doesn't use any of the random choice above
    "gg_gg":       [ ("gg_gg", lambda ids: ids[:4]) ],
    "gg_qqx":      [ ("gg_qqx", lambda ids: ids[:4]) ],
    "gq_gq":       [ ("gq_gq", lambda ids: ids[:4]) ],
    "gqx_gqx":     [ ("gqx_gqx", lambda ids: ids[:4]) ],
    "qq_qq":       [ ("qq_qq", lambda ids: ids[:4]) ],
    "qqp_qqp":     [ ("qqp_qqp", lambda ids: ids[:4]) ],
    "qxqx_qxqx":   [ ("qxqx_qxqx", lambda ids: ids[:4]) ],
    "qxqpx_qxqpx": [ ("qxqpx_qxqpx", lambda ids: ids[:4]) ],
    "qqpx_qqpx":   [ ("qqpx_qqpx", lambda ids: ids[:4]) ],
    "qqx_gg":      [ ("qqx_gg", lambda ids: ids[:4]) ],
    "qqx_qqx":     [ ("qqx_qqx",lambda ids: ids[:4]) ],
    "qqx_qpqpx":   [ ("qqx_qpqpx", lambda ids: ids[:4]) ],
}


IDmap_2to3 = {
    #Parton ID maps for 2->3 processes. 
    #These take in [ID1, ID2, outIDs]
    
    # (1) gg → gg
    "gg_gg": [
        ("gg_ggg", lambda incoming: [21, 21, 21, 21, 21]),
        ("gg_qqxg", lambda incoming: [21, 21] + random_quark_pair() + [21]),
    ],
    
    # (2) gg → qq̄
    "gg_qqx": [
        ("gg_qqxg", lambda incoming: [21, 21] + incoming[2:] + [21]), 
    ],

    # (3) gq → gq
    "gq_gq": [
        ("gq_gqg", lambda incoming: incoming[:2] + incoming[:2] + [21]),
        ("gq_qqxq", lambda incoming: incoming[:2] + [incoming[1], -incoming[1], incoming[1]]),
        ("gq_qpqpxq", lambda incoming: incoming[:2] + random_new_quark_pair([abs(incoming[1])]) + [incoming[1]]),
    ],

    # (4) gq̄ → gq̄
    "gqx_gqx": [
        ("gqx_gqxg", lambda incoming: incoming[:2] + incoming[:2] + [21]),
        ("gqx_qqxqx", lambda incoming: incoming[:2] + [abs(incoming[1]), -abs(incoming[1]), -abs(incoming[1])]),
        ("gqx_qpqpxqx", lambda incoming: incoming[:2] + random_new_quark_pair([abs(incoming[1])]) + [incoming[1]]),
    ],
    
    # (5) qq → qq 
    "qq_qq": [
        ("qq_qqg", lambda incoming: incoming[:2] * 2 + [21]),
    ],

    # (6) qq' → qq' 
    "qqp_qqp": [
        ("qqp_qqpg", lambda incoming: incoming[:2] * 2 + [21]),
    ],
    
    # (7) q̄q̄ → q̄q̄ 
    "qxqx_qxqx": [
        ("qxqx_qxqxg", lambda incoming: incoming[:2] * 2 + [21]),
    ],
    
    # (8) q̄q̄' → q̄q̄' 
    "qxqpx_qxqpx": [
        ("qxqpx_qxqpxg", lambda incoming: incoming[:2] * 2 + [21]),
    ],
    
    # (9) qq̄' → qq̄'
    "qqpx_qqpx": [
        ("qqpx_qqpxg", lambda incoming: incoming[:2] * 2 + [21]),
    ],   
    
    # (10) qq̄ → gg
    "qqx_gg": [
        ("qqx_ggg", lambda incoming: incoming[:2] + [21, 21, 21]),
        ("qqx_qqxg", lambda incoming: incoming[:2] + incoming[:2] + [21]),
        ("qqx_qpqpxg", lambda incoming: incoming[:2] + random_new_quark_pair(incoming[:2]) + [21]),
    ],
    
    # (12) qq̄ → qq̄ 
    "qqx_qqx": [
        ("qqx_qqxg", lambda incoming: incoming[:2] * 2 + [21]),
    ],
    
    # (11) qq̄ → q'q̄' 
    "qqx_qpqpx": [
        ("qqx_qpqpxg", lambda incoming: incoming[:2] + random_new_quark_pair(incoming[:2]) + [21]),
    ], 
}


IDmap_2to4 = {
    #Parton ID maps for 2->4 processes. 
    #These take in [ID1, ID2, outIDs]
    
    # (1) gg → gg
    "gg_gg": [
        ("gg_gggg", lambda incoming: [21, 21, 21, 21, 21, 21]),
        ("gg_ggqqx", lambda incoming: [21, 21] + random_quark_pair() + [21, 21]),
        ("gg_qqxqqx", lambda incoming: (lambda pair=random_quark_pair(): [21, 21] + pair + pair)()),
        ("gg_qqxqpqpx", lambda incoming: [21, 21] + random_qqxqpqpx_ids()),
    ],
    
    # (2) gg → qq̄
    "gg_qqx": [
        ("gg_ggqqx", lambda incoming: [21, 21] + incoming[2:] + [21, 21]),  
        ("gg_qqxqqx", lambda incoming: [21, 21] + incoming[2:] + incoming[2:]), 
        ("gg_qqxqpqpx", lambda incoming: [21, 21] + incoming[2:] + random_new_quark_pair(incoming[2:])),  
    ],

    # (3) gq → gq
    "gq_gq": [
        ("gq_gqgg", lambda incoming: incoming[:2] + incoming[:2] + [21, 21]),
        ("gq_gqqqx", lambda incoming: incoming[:2] + incoming[:2] + [incoming[1], -incoming[1]]),
        ("gq_gqqpqpx", lambda incoming: incoming[:2] + incoming[:2] + random_new_quark_pair([incoming[:2]])),
    ],

    # (4) gq̄ → gq̄
    "gqx_gqx": [
        ("gqx_gqxgg", lambda incoming: incoming[:2] + incoming[:2] + [21, 21]),
        ("gqx_gqxqqx", lambda incoming: incoming[:2] + incoming[:2] + [abs(incoming[1]), -abs(incoming[1])]),
        ("gqx_gqxqpqpx", lambda incoming: incoming[:2] + incoming[:2] + random_new_quark_pair([abs(incoming[1])])),
    ],
    
    # (5) qq → qq 
    "qq_qq": [
        ("qq_qqgg", lambda incoming: incoming[:2] * 2 + [21, 21]),
        ("qq_qqqqx", lambda incoming: incoming[:2] * 2 + [incoming[0], -incoming[0]]),
        ("qq_qqqpqpx", lambda incoming: incoming[:2] * 2 + random_new_quark_pair(incoming[:2])),
    ],

    # (6) qq' → qq' 
    "qqp_qqp": [
        ("qqp_qqpgg", lambda incoming: incoming[:2] * 2 + [21, 21]),
        ("qqp_qqpqqx", lambda incoming: incoming[:2] * 2 + [incoming[0], -incoming[0]]),
        ("qqp_qqpqpqpx", lambda incoming: incoming[:2] * 2 + [incoming[1], -incoming[1]]),
        ("qqp_qqpoox", lambda incoming: incoming[:2] * 2 + random_new_quark_pair(incoming[:2])),
    ],
    
    # (7) q̄q̄ → q̄q̄ 
    "qxqx_qxqx": [
        ("qxqx_qxqxgg", lambda incoming: incoming[:2] * 2 + [21, 21]),
        ("qxqx_qxqxqqx", lambda incoming: incoming[:2] * 2 + [incoming[0], -incoming[0]]),
        ("qxqx_qxqxqpqpx", lambda incoming: incoming[:2] * 2 + random_new_quark_pair([abs(x) for x in incoming[:2]])),
    ],
    
    # (8) q̄q̄' → q̄q̄' 
    "qxqpx_qxqpx": [
        ("qxqpx_qxqpxgg", lambda incoming: incoming[:2] * 2 + [21, 21]),
        ("qxqpx_qxqpxqqx", lambda incoming: incoming[:2] * 2 + [incoming[0], -incoming[0]]),
        ("qxqpx_qxqpxqpqpx", lambda incoming: incoming[:2] * 2 + [incoming[1], -incoming[1]]),
        ("qxqpx_qxqpxoox", lambda incoming: incoming[:2] * 2 + random_new_quark_pair([abs(x) for x in incoming[:2]])),
    ],
    
    # (9) qq̄' → qq̄'
    "qqpx_qqpx": [
        ("qqpx_qqpxgg", lambda incoming: incoming[:2] * 2 + [21, 21]),
        ("qqpx_qqpxqqx", lambda incoming: incoming[:2] * 2 + [incoming[0], -incoming[0]]),
        ("qqpx_qqpxqpqpx", lambda incoming: incoming[:2] * 2 + [abs(incoming[1]), -abs(incoming[1])]),
        ("qqpx_qqpxoox", lambda incoming: incoming[:2] * 2 + random_new_quark_pair([abs(x) for x in incoming[:2]])),
    ],    
    

    # (10) qq̄ → gg
    "qqx_gg": [
        ("qqx_gggg", lambda incoming: incoming[:2] + [21, 21, 21, 21]),
        ("qqx_ggqqx", lambda incoming: incoming[:2] + [21, 21] + incoming[:2]),
        ("qqx_ggqpqpx", lambda incoming: incoming[:2] + [21, 21] + random_new_quark_pair(incoming[:2])),
        ("qqx_qqxqqx", lambda incoming: incoming[:2] * 3),
        ("qqx_qqxqpqpx", lambda incoming: incoming[:2] * 2 + random_new_quark_pair(incoming[:2])),
        ("qqx_qpqpxqpqpx", lambda incoming: incoming[:2] + random_new_quark_pair(incoming[:2]) * 2),
        ("qqx_qpqpxoox", lambda incoming: incoming[:2] + random_new_quark_pair(incoming[:2]) + random_new_quark_pair(incoming[:2])),
    ],
    
    # (11) qq̄ → qq̄
    "qqx_qqx": [
        ("qqx_qqxgg", lambda incoming: incoming[:2] * 2 + [21, 21]),
        ("qqx_qqxqqx", lambda incoming: incoming[:2] * 2 + [incoming[0], -incoming[0]]),
        ("qqx_qqxqpqpx", lambda incoming: incoming[:2] * 2 + random_new_quark_pair([abs(x) for x in incoming[:2]])),
    ],
    
    # (12) qq̄ → q'q̄' 
    "qqx_qpqpx": [
        ("qqx_qpqpxgg", lambda incoming: incoming[:2] + random_new_quark_pair(incoming[:2]) + [21, 21]),
        ("qqx_qqxqpqpx", lambda incoming: incoming[:2] * 2 + random_new_quark_pair(incoming[:2])),
        ("qqx_qpqpxqpqpx", lambda incoming: incoming[:2] + random_new_quark_pair(incoming[:2]) * 2),
        ("qqx_qpqpxoox", lambda incoming: incoming[:2] + random_new_quark_pair(incoming[:2]) + random_new_quark_pair(incoming[:2])),
    ],
}


IDmap_2to5 = {
    #Parton ID maps for 2->5 processes. 
    #These take in [ID1, ID2, outIDs]
    
    # (1) gg → gg
    "gg_gg": [
        # ("gg_ggggg", lambda incoming: [21, 21, 21, 21, 21, 21, 21]), # MG did not generate any events of this 
        ("gg_ggqqxg", lambda incoming: [21, 21, 21] + random_quark_pair() + [21, 21]),
        ("gg_qqxqqxg", lambda incoming: (lambda pair=random_quark_pair(): [21, 21] + pair + pair)() + [21]),
        ("gg_qqxqpqpxg", lambda incoming: [21, 21] + random_qqxqpqpx_ids() + [21]),
    ],
    
    # (2) gg → qq̄
    "gg_qqx": [
        ("gg_ggqqxg", lambda incoming: [21, 21] + incoming[2:] + [21, 21, 21]),  
        ("gg_qqxqqxg", lambda incoming: [21, 21] + incoming[2:] + incoming[2:] + [21]), 
        ("gg_qqxqpqpxg", lambda incoming: [21, 21] + incoming[2:] + random_new_quark_pair(incoming[2:]) + [21]),  
    ],

    # (3) gq → gq
    "gq_gq": [
        ("gq_gqggg", lambda incoming: incoming[:2] + incoming[:2] + [21, 21, 21]),
        ("gq_gqqqxg", lambda incoming: incoming[:2] + incoming[:2] + [incoming[1], -incoming[1]] + [21]),
        ("gq_gqqpqpxg", lambda incoming: incoming[:2] + incoming[:2] + random_new_quark_pair([incoming[:2]]) + [21]),
        ("gq_qqxqqxq", lambda incoming: incoming[:2] + [incoming[1], -incoming[1], incoming[1], -incoming[1], incoming[1]]), 
        ("gq_qqxqpqpxq", lambda incoming: incoming[:2] + [incoming[1], -incoming[1]] + random_new_quark_pair([incoming[1]]) + [incoming[1]]),
        ("gq_qpqpxqpqpxq", lambda incoming: incoming[:2] + random_new_quark_pair(incoming[:2]) * 2 + [incoming[1]]),
        ("gq_qpqpxooxq", lambda incoming: incoming[:2] + random_new_quark_pair(incoming[:2]) + random_new_quark_pair([incoming[:2]]) + [incoming[1]]),
    ],

    # (4) gq̄ → gq̄
    "gqx_gqx": [
        ("gqx_gqxggg", lambda incoming: incoming[:2] + incoming[:2] + [21, 21, 21]),
        ("gqx_gqxqqxg", lambda incoming: incoming[:2] + incoming[:2] + [abs(incoming[1]), -abs(incoming[1])] + [21]),
        ("gqx_gqxqpqpxg", lambda incoming: incoming[:2] + incoming[:2] + random_new_quark_pair([abs(incoming[1])]) + [21]),
        ("gqx_qqxqqxqx", lambda incoming: incoming[:2] + [-incoming[1], incoming[1], -incoming[1], incoming[1]] + [-abs(incoming[1])]),
        ("gqx_qpqpxqqxqx", lambda incoming: incoming[:2] + [incoming[1], -incoming[1]] + random_new_quark_pair(incoming[:2]) + [-abs(incoming[1])]),
        ("gqx_qpqpxqpqpxqx", lambda incoming: incoming[:2] + random_new_quark_pair(incoming[:2]) * 2 + [-abs(incoming[1])]),
        ("gqx_qpqpxooxqx", lambda incoming: incoming[:2] + random_new_quark_pair(incoming[:2]) + random_new_quark_pair(incoming[:2]) + [-abs(incoming[1])]),
    ],
    
    # (5) qq → qq 
    "qq_qq": [
        ("qq_qqggg", lambda incoming: incoming[:2] * 2 + [21, 21, 21]),
        ("qq_qqqqxg", lambda incoming: incoming[:2] * 2 + [incoming[0], -incoming[0]] + [21]),
        ("qq_qqqpqpxg", lambda incoming: incoming[:2] * 2 + random_new_quark_pair(incoming[:2]) + [21]),
    ],

    # (6) qq' → qq' 
    "qqp_qqp": [
        ("qqp_qqpggg", lambda incoming: incoming[:2] * 2 + [21, 21, 21]),
        ("qqp_qqpqqxg", lambda incoming: incoming[:2] * 2 + [incoming[0], -incoming[0]] + [21]),
        ("qqp_qqpqpqpxg", lambda incoming: incoming[:2] * 2 + [incoming[1], -incoming[1]] + [21]),
        ("qqp_qqpooxg", lambda incoming: incoming[:2] * 2 + random_new_quark_pair(incoming[:2]) + [21]),
    ],
    
    # (7) q̄q̄ → q̄q̄ 
    "qxqx_qxqx": [
        ("qxqx_qxqxggg", lambda incoming: incoming[:2] * 2 + [21, 21, 21]),
        ("qxqx_qxqxqqxg", lambda incoming: incoming[:2] * 2 + [incoming[0], -incoming[0]] + [21]),
        ("qxqx_qxqxqpqpxg", lambda incoming: incoming[:2] * 2 + random_new_quark_pair([abs(x) for x in incoming[:2]]) + [21]),
    ],
    
    # (8) q̄q̄' → q̄q̄' 
    "qxqpx_qxqpx": [
        ("qxqpx_qxqpxggg", lambda incoming: incoming[:2] * 2 + [21, 21, 21]),
        ("qxqpx_qxqpxqqxg", lambda incoming: incoming[:2] * 2 + [incoming[0], -incoming[0]] + [21]),
        ("qxqpx_qxqpxqpqpxg", lambda incoming: incoming[:2] * 2 + [incoming[1], -incoming[1]] + [21]),
        ("qxqpx_qxqpxooxg", lambda incoming: incoming[:2] * 2 + random_new_quark_pair([abs(x) for x in incoming[:2]]) + [21]),
    ],
    
    # (9) qq̄' → qq̄'
    "qqpx_qqpx": [
        ("qqpx_qqpxggg", lambda incoming: incoming[:2] * 2 + [21, 21, 21]),
        ("qqpx_qqpxqqxg", lambda incoming: incoming[:2] * 2 + [incoming[0], -incoming[0]] + [21]),
        ("qqpx_qqpxqpqpxg", lambda incoming: incoming[:2] * 2 + [abs(incoming[1]), -abs(incoming[1])] + [21]),
        ("qqpx_qqpxooxg", lambda incoming: incoming[:2] * 2 + random_new_quark_pair([abs(x) for x in incoming[:2]]) + [21]),
    ],    
    
    # (10) qq̄ → gg
    "qqx_gg": [
        ("qqx_ggggg", lambda incoming: incoming[:2] + [21, 21, 21, 21, 21]),
        ("qqx_ggqqxg", lambda incoming: incoming[:2] + [21, 21] + incoming[:2] + [21]),
        ("qqx_ggqpqpxg", lambda incoming: incoming[:2] + [21, 21] + random_new_quark_pair(incoming[:2]) + [21]),
        ("qqx_qqxqqxg", lambda incoming: incoming[:2] * 3 + [21]),
        ("qqx_qqxqpqpxg", lambda incoming: incoming[:2] * 2 + random_new_quark_pair(incoming[:2]) + [21]),
        ("qqx_qpqpxqpqpxg", lambda incoming: incoming[:2] + random_new_quark_pair(incoming[:2]) * 2 + [21]),
        ("qqx_qpqpxooxg", lambda incoming: incoming[:2] + random_new_quark_pair(incoming[:2]) + random_new_quark_pair(incoming[:2]) + [21]),
    ],
    
    # (11) qq̄ → qq̄ 
    "qqx_qqx": [
        ("qqx_qqxggg", lambda incoming: incoming[:2] * 2 + [21, 21, 21]),
        ("qqx_qqxqqxg", lambda incoming: incoming[:2] * 2 + [incoming[0], -incoming[0]] + [21]),
        ("qqx_qqxqpqpxg", lambda incoming: incoming[:2] * 2 + random_new_quark_pair([abs(x) for x in incoming[:2]]) + [21]),
    ],
    
    # (12) qq̄ → q'q̄' 
    "qqx_qpqpx": [
        ("qqx_qpqpxggg", lambda incoming: incoming[:2] + random_new_quark_pair(incoming[:2]) + [21, 21, 21]),
        ("qqx_qqxqpqpxg", lambda incoming: incoming[:2] * 2 + random_new_quark_pair(incoming[:2]) + [21]),
        ("qqx_qpqpxqpqpxg", lambda incoming: incoming[:2] + random_new_quark_pair(incoming[:2]) * 2 + [21]),
        ("qqx_qpqpxooxg", lambda incoming: incoming[:2] + random_new_quark_pair(incoming[:2]) + random_new_quark_pair(incoming[:2]) + [21]),
    ],

}



#=================================================================
#=================== COLOUR FLOW HELPER  =========================
#=================================================================

def parse_flows(file_path):
    # Look into colour flow folder and get all with weights
    flows = []  # Each item: (weight, [(a,b), (c,d), ...])
    
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("Set:"):
                continue
            
            # Extract weight after "Seen:"
            weight_match = re.search(r"Seen:\s*(\d+)", line)
            weight = int(weight_match.group(1)) if weight_match else 1
            
            # Extract all pairs like (501,503)
            pairs = re.findall(r"\((\d+),(\d+)\)", line)
            colour_pairs = [(int(a), int(b)) for a, b in pairs]
            
            flows.append((weight, colour_pairs))
    
    return flows


def select_weighted_flow(flows):
    # Select colour flow based on weight
    # flows = [(weight, [(a,b), (c,d), ...]), ...]
    total_weight = sum(weight for weight, _ in flows)
    r = random.uniform(0, total_weight)
    
    cumulative = 0
    for weight, pairs in flows:
        cumulative += weight
        if r <= cumulative:
            return pairs
        

def assign_ids_from_colourflow(chosen_colour_flow, FullIDs):
    # Assign/reorder PDG IDs to match the colour-flow pattern.
    # chosen_colour_flow: list of tuples [(c1,c2), ...]
    # FullIDs: list of PDG IDs (first 2 = incoming, rest = outgoing)
    # Returns: FullIDs_from_flow: reordered list of IDs matching colour-flow types
    if len(chosen_colour_flow) != len(FullIDs):
        raise ValueError("Colour flow length doesn't match number of particles")

    FullIDs_from_flow = []

    # Helper type checks
    def is_gluon(pid): return abs(pid) == 21
    def is_quark(pid): return 1 <= pid <= 6
    def is_antiquark(pid): return -6 <= pid <= -1

    # Separate flows by structure
    gluon_flows = [f for f in chosen_colour_flow if f[0] != 0 and f[1] != 0]
    quark_flows = [f for f in chosen_colour_flow if f[0] != 0 and f[1] == 0]
    antiquark_flows = [f for f in chosen_colour_flow if f[0] == 0 and f[1] != 0]

    # Separate IDs by type
    gluons = [pid for pid in FullIDs if is_gluon(pid)]
    quarks = [pid for pid in FullIDs if is_quark(pid)]
    antiquarks = [pid for pid in FullIDs if is_antiquark(pid)]

    # Build reordered ID list based on flow structure
    for (c1, c2) in chosen_colour_flow:
        if c1 != 0 and c2 != 0:  # gluon flow
            if not gluons:
                raise ValueError("No gluon ID available for gluon flow")
            FullIDs_from_flow.append(gluons.pop(0))
        elif c1 != 0 and c2 == 0:  # quark flow
            if not quarks:
                raise ValueError("No quark ID available for quark flow")
            FullIDs_from_flow.append(quarks.pop(0))
        elif c1 == 0 and c2 != 0:  # antiquark flow
            if not antiquarks:
                raise ValueError("No antiquark ID available for antiquark flow")
            FullIDs_from_flow.append(antiquarks.pop(0))
        else:
            # uncoloured (shouldn’t happen in QCD)
            FullIDs_from_flow.append(0)

    return FullIDs_from_flow
    
    

#=================================================================
#========================= QCD FUNCTIONS =========================
#=================================================================


def QCD(parameters, M, ID1, ID2, s, PDF, MinMass, MaxMass, MSquaredFunc):
    # Get QCD differential cross section for integration
    Y, y = parameters
    PDFScale = M

    # Basic sanity checks
    if M <= 0 or s <= 0:
        raise ValueError(f"Invalid kinematics: M={M}, s={s}")

    tau = (M**2) / s

    # Parton momentum fractions
    xa = np.sqrt(tau) * np.exp(Y)
    xb = np.sqrt(tau) * np.exp(-Y)
    if xa <= 0 or xb <= 0 or xa > 1.0 or xb > 1.0:
        raise ValueError(f"Invalid momentum fractions: xa={xa}, xb={xb}, Y={Y}, tau={tau}")

    alpha3 = 1.0 / (1.0 / 0.118 + 7.0 / (2.0 * np.pi) * np.log(PDFScale / 91.2))
    g3 = np.sqrt(max(0.0, 4.0 * np.pi * alpha3))

    shat = M**2
    cosh_y = np.cosh(y)
    if cosh_y == 0:
        raise ValueError("cosh_y is 0")
    that = -0.5 * M**2 * np.exp(-y) / cosh_y
    uhat = -0.5 * M**2 * np.exp(y)  / cosh_y

    # # Standard minimum pT cut for QCD backgrounds. Required to stop divergences in cross sections
    # This works, but then the generation takes forever
    # Better to just use a ymax cut in the configs area 
    # pT2 = (that * uhat) / shat # pT squared
    # pT_min = 10.0 # GeV
    # if pT2 < pT_min**2:
    #     return 0.0, xa, xb

    # Matrix Element
    MSquareQCD = g3**4 * MSquaredFunc(shat, that, uhat)

    # Prefactor (avoid division by zero)
    denom = (np.cosh(y)**2 * 16.0 * np.pi * shat**2)
    if denom == 0:
        raise ValueError(f"Denom is 0. xa={xa}, xb={xb}, Y={Y}, tau={tau}")

    prefactor = MSquareQCD * M / denom

    # PDF evaluation: guard LHAPDF input types
    try:
        PDFvalue = float(PDF.xfxQ(int(ID1), float(xa), PDFScale)) * float(PDF.xfxQ(int(ID2), float(xb), PDFScale))
    except Exception:
        raise ValueError(f"PDF call failed")

    # Beam swap term (if incoming particles are different)
    if ID1 != ID2:
        try:
            PDFvalue += float(PDF.xfxQ(int(ID1), float(xb), PDFScale)) * float(PDF.xfxQ(int(ID2), float(xa), PDFScale))
        except Exception:
            pass

    convol = PDFvalue * prefactor
    
    if not np.isfinite(convol):
        raise ValueError(f"Convolution is not finite. xa={xa}, xb={xb}, Y={Y}, tau={tau}")

    return convol, xa, xb


def convolution(parameters, M, ID1, ID2, s, PDF, MinMass, MaxMass, MSquaredFunc):
    result, _, _ = QCD(parameters, M, ID1, ID2, s, PDF, MinMass, MaxMass, MSquaredFunc)
    return result


def xs_and_colour(MinMass, MaxMass, M, s, yMax, PDF, MC, MSquaredFunc, ID1, ID2, outIDs=None, maxTries=2000):
    # Get (xa, xb, ID1, ID2, outIDs, Y, y) for a single event 
    for _ in range(maxTries):
        if outIDs is None:
            outIDs = PartonOutIDs(MSquaredFunc, ID1, ID2)

        tau = M**2 / s
        Ymax_M = min(np.log(1 / np.sqrt(tau)), yMax)
        RandomY = np.random.uniform(-Ymax_M + 1e-9, Ymax_M - 1e-9)
        Randomy = np.random.uniform(-(yMax - abs(RandomY)), yMax - abs(RandomY))

        MaximumAmplitude, _, _ = QCD([0, 0], M, ID1, ID2, s, PDF, MinMass, MaxMass, MSquaredFunc)
        MaximumAmplitude *= 1.2
        ActualAmplitude, xa, xb = QCD([RandomY, Randomy], M, ID1, ID2, s, PDF, MinMass, MaxMass, MSquaredFunc)

        if np.random.uniform(0.0, MaximumAmplitude) < ActualAmplitude:
            return xa, xb, ID1, ID2, outIDs, RandomY, Randomy

    raise Exception(f"Failed to sample kinematics after {maxTries} tries.")


def QCD_envelope(parameters, M, ID1, ID2, s, MSquaredFunc):
    Y, y = parameters
    PDFScale = M

    shat = M**2
    cosh_y = np.cosh(y)

    that = -0.5 * M**2 * np.exp(-y) / cosh_y
    uhat = -0.5 * M**2 * np.exp(y)  / cosh_y

    alpha3 = 1.0 / (1.0 / 0.118 + 7.0 / (2.0 * np.pi) * np.log(PDFScale / 91.2))
    g3 = np.sqrt(max(0.0, 4.0 * np.pi * alpha3))

    return g3**4 * MSquaredFunc(shat, that, uhat)

def subprocess_envelopes(M, yMax, s, MC, all_subprocesses, subprocess_combinations, PDF):
    envelope_maxes = []

    for subprocess in all_subprocesses:
        max_val = 0.0

        for ID1, ID2, MSquaredFunc in subprocess_combinations(subprocess):

            # sample a few random points instead of integrating
            for _ in range(MC):

                Ymax_M = min(np.log(1 / np.sqrt(M**2 / s)), yMax)

                Y = np.random.uniform(-Ymax_M, Ymax_M)
                y = np.random.uniform(-(yMax - abs(Y)), yMax - abs(Y))

                w, _, _ = QCD((Y, y), M, ID1, ID2, s, PDF, 0, 0, MSquaredFunc)
                # w = QCD_envelope((Y, y), M, ID1, ID2, s, MSquaredFunc)

                if w > max_val:
                    max_val = w

        envelope_maxes.append(max_val)

    return envelope_maxes


def subprocess_xsecs(M, yMax, s, MC, Subprocesses, subprocess_combinations, PDF, MinMass, MaxMass):
    # Calculate the cross section of each process and general info
    tau = M**2 / s
    Ymax_M = min(np.log(1 / np.sqrt(tau)), yMax)
    
    cross_sections = []
    process_info = []

    for (subprocess, MSquaredFunc, lprup) in Subprocesses:
        for ID1, ID2, _ in subprocess_combinations(subprocess):
            # Integrate via the convolution wrapper just like subprocess_envelopes. This gives 𝑑𝜎/dM
            sum_at_M = Integrate(
                convolution, 
                (M, ID1, ID2, s, PDF, MinMass, MaxMass, MSquaredFunc),
                MC, -Ymax_M, Ymax_M, yMax
            )
            
            cross_sections.append(sum_at_M)
            process_info.append((subprocess, ID1, ID2, MSquaredFunc, lprup))

    return cross_sections, process_info


#=======================================================================
#========================= GENERATE EVENTS =============================
#=======================================================================

def generate_events(dimensionality, min_mass_TeV, max_mass_TeV, N_events, params, dirs, process_map):
    # Generate QCD or massless phase space events using CoM dynamics 
    # Boosts to lab frame and store in LHE files 
    
    MinMass = min_mass_TeV * params['TeV2GeV']
    MaxMass = max_mass_TeV * params['TeV2GeV']
    output_type = params['output_type']
    s      = params['s']
    yMax   = params['yMax']
    PDF    = params['PDF']
    MC     = params['MC']
    Npartons  = params['Npartons']
    
    if Npartons == 2:
        output_type = params["output_type"]   # "PS" or "QCD"
    else:
        output_type = "PS"                    # Default for Npartons != 2

    if output_type not in ["PS", "QCD"]:
        raise ValueError(f"Invalid mode: {output_type}")

    active_names = [name for name, (_, _, _, active) in process_map.items() if active]

    M_ref = 0.5 * (MinMass + MaxMass)
    envelope_maxes = subprocess_envelopes(M_ref, yMax, s, MC, active_names, subprocess_combinations, PDF)
    
    # Bookkeeping 
    N_subproc      = [0]*len(active_names)   # Counts events per subprocess
    Weight_subproc = [0.0]*len(active_names) # Sum of weights per subprocess
    Error_subproc  = [0.0]*len(active_names) # Sum of squared weights per subprocess
    eventsLab = []
    interaction_counter = Counter()
    full_process_counter = Counter()
    fail_count = 0
    accepted_count = 0  # Accept exactly N_events
    
    DeltaMass = (MaxMass - MinMass) / N_events

    # Main loop: keep going until we have exactly N_events accepted 
    while accepted_count < N_events:
        CrossSections = []
        
        try:            
            # 1) Sample invariant mass in [MinMass, MaxMass]
            M = random.uniform(MinMass, MaxMass)
            if M >= s: continue  # pathological, resample

            # 2) Cross section at this M 
            Subprocesses = [
                (name, globals()[func_name], lprup)
                for name, (folder, func_name, lprup, active) in process_map.items()
                if active
            ]
            CrossSections, ProcessInfo = subprocess_xsecs(
                M, yMax, s, MC, Subprocesses, subprocess_combinations, PDF, MinMass, MaxMass
            )

            CrossSectionsSum = np.cumsum(CrossSections)
            TotalXSec = CrossSectionsSum[-1]
            event_xsec = TotalXSec * DeltaMass
            
            # 3) Pick subprocess index based on cross section weights
            RSig = np.random.uniform(0, TotalXSec)
            InteractionIndex = np.searchsorted(CrossSectionsSum, RSig, side='left')

            interaction_name, ID1, ID2, MSquaredFunc, lprup = ProcessInfo[InteractionIndex]

            env_index = active_names.index(interaction_name)

            RCrossSection = random.uniform(0, envelope_maxes[env_index])
            if RCrossSection > CrossSections[InteractionIndex]:
                continue
                
            # 4) Sample kinematics
            xa, xb, ID1, ID2, outIDs, RandomY, Randomy = xs_and_colour(
                MinMass, MaxMass, M, s, yMax, PDF, MC, MSquaredFunc, ID1, ID2
            )            

            # 5) Event weight (per-event differential weight at M, in pb) 
            # Convert GeV^-2 to picobarns
            conversion_pb = 0.389379e9                
            weight = event_xsec * conversion_pb
            subproc_idx = env_index

            N_subproc[subproc_idx]      += 1
            Weight_subproc[subproc_idx] += weight
            Error_subproc[subproc_idx]  += weight**2

            # 6) Build IDs + Colour Flow
            incoming_ids = [ID1, ID2] + outIDs
            subprocess_folder = process_map[interaction_name][0]

            # Map Npartons → corresponding ID map
            IDMAPS = {
                2: IDmap_2to2,
                3: IDmap_2to3,
                4: IDmap_2to4,
                5: IDmap_2to5,
            }

            if Npartons not in IDMAPS:
                raise ValueError(f"Unsupported final-state multiplicity: {Npartons}")

            idmap = IDMAPS[Npartons]

            if interaction_name not in idmap or not idmap[interaction_name]:
                print(f"[ERROR] No ID map entries for {interaction_name} (Npartons={Npartons})")
                continue

            # Randomly pick one subprocess mapping
            selected_process = random.choice(idmap[interaction_name])
            process_name, id_generator = selected_process
            FullIDs = id_generator(incoming_ids)

            two_to = f"2to{Npartons}"
            look_here = process_name  # e.g. "gg_gg", "gg_ggg", "gg_gggg" etc.

            # 6.5 Colour-flow lookup
            cf_dir = os.path.join("ColourFlow", "MadGraph", two_to, subprocess_folder)
            chosen_colour_flow = None

            if os.path.isdir(cf_dir):
                target_file = os.path.join(cf_dir, f"{look_here}.txt")
                if os.path.isfile(target_file):
                    flows = parse_flows(target_file)
                    if flows:
                        chosen_colour_flow = select_weighted_flow(flows)
                        #print(f"[INFO] Using colour flow from: {target_file}")
                    else:
                        print(f"[ERROR] No flows parsed from {target_file}")
                else:
                    print(f"[ERROR] Missing colour flow file: {target_file}")
            else:
                print(f"[ERROR] Missing colour flow directory: {cf_dir}")

            if chosen_colour_flow is None:
                print(f"[ERROR] No valid colour flow found for {interaction_name} ({look_here}) in {cf_dir}")
                continue
                
            FullIDs_from_flow = assign_ids_from_colourflow(chosen_colour_flow, FullIDs)

            # 7) Kinematics  
            pIn_CoM = [
                (M/2, 0, 0, M/2),
                (M/2, 0, 0, -M/2)
            ]

            if params['Npartons'] in [3,4,5]:
                # Always phase space for 2→3,4,5
                if dimensionality == "3D": 
                    outgoing_particles_CoM = phase3(params['Npartons'], M) 
                elif dimensionality == "2D":
                    random_choice = np.random.binomial(1, params['DR_prob']) if params['DR_flag'] else 0
                    outgoing_particles_CoM = phase3(params['Npartons'], M) if random_choice else phase2(params['Npartons'], M)
                else:
                    raise ValueError(f"Unsupported dimensionality={dimensionality} for Npartons={params['Npartons']}")

                full_event_CoM = np.vstack((pIn_CoM, outgoing_particles_CoM))

            elif params['Npartons'] == 2:
                if output_type == "PS":
                    # 2→2 Phase space
                    if dimensionality == "3D": 
                        outgoing_particles_CoM = phase3(params['Npartons'], M) 
                    elif dimensionality == "2D":
                        random_choice = np.random.binomial(1, params['DR_prob']) if params['DR_flag'] else 0
                        outgoing_particles_CoM = phase3(params['Npartons'], M) if random_choice else phase2(params['Npartons'], M)
                    else:
                        raise ValueError(f"Unsupported dimensionality={dimensionality} for Npartons={params['Npartons']}")

                    full_event_CoM = np.vstack((pIn_CoM, outgoing_particles_CoM))

                elif output_type == "QCD":
                    # 2→2 QCD kinematics 
                    outgoing_particles_CoM, pt = QCD_2to2_outgoingkin(
                        interaction_name, M, RandomY, Randomy, outIDs, QuarkMasses
                    )
                    full_event_CoM = np.vstack((pIn_CoM, outgoing_particles_CoM))

                else:
                    raise ValueError(f"Unsupported mode={output_type} for Npartons=2")

            else:
                raise ValueError(f"Unsupported Npartons={params['Npartons']}")

            # Check first that events in the CoM are fine (they should always be)
            if not check_event_physical(full_event_CoM, label="CoM"):
                fail_count += 1
                continue

            # Incoming 4-vectors in lab
            pIn_lab = np.array([
                [xa * params['Ebeam'], 0, 0,  xa * params['Ebeam']],
                [xb * params['Ebeam'], 0, 0, -xb * params['Ebeam']],
            ])
            p_tot_lab = pIn_lab.sum(axis=0)
            beta_lab = p_tot_lab[1:] / p_tot_lab[0]  # velocity of CoM in lab frame
            gamma_lab = 1.0 / np.sqrt(1 - np.dot(beta_lab, beta_lab))

            full_event_Lab = np.array([boost(gamma_lab, beta_lab, p) for p in full_event_CoM])

            # Check the lab events, sometimes unphyscial after boost...
            if not check_event_physical(full_event_Lab, label="Lab"):
                fail_count += 1
                continue
                
            # Append only if the boost works and only in the lab frame (LHE are only lab frame events)
            eventsLab.append((
                full_event_Lab, xa, xb, M, ID1, ID2, FullIDs_from_flow,
                InteractionIndex, lprup, selected_process, chosen_colour_flow
            ))
                    
            # 8) Update Counters
            interaction_counter[interaction_name] += 1
            full_process_counter[(interaction_name, (ID1, ID2), selected_process, tuple(FullIDs))] += 1

            accepted_count += 1
            if (accepted_count % 500 == 0) and (accepted_count < N_events):
                print(f"[INFO] Generated {accepted_count}/{N_events} events...", end='\r', flush=True)

        except Exception as e:
            print(f"[ERROR] Event attempt failed: {e}")
            fail_count += 1
            break

    # Only include active processes
    active_processes = [(name, folder, ME, lprup) 
                        for name, (folder, ME, lprup, active) in process_map.items() if active]

    CrossSections_list = []
    lprup_by_index = [lprup for _, _, _, lprup in active_processes]

    for i, (name, folder, ME, lprup) in enumerate(active_processes):
        if N_subproc[i] > 0:
            xsec = Weight_subproc[i]
            err  = math.sqrt(Error_subproc[i]) / N_events
        else:
            xsec, err = 0.0, 0.0
        CrossSections_list.append((xsec, err, 0.0, lprup))

    # Write LHE files
    WriteLHE(eventsLab, dirs['lab_file'], s, PDF, CrossSections_list, lprup_by_index)

    print(f"✔ Finished {min_mass_TeV:.2f}–{max_mass_TeV:.2f} TeV | "
          f"Events: {accepted_count} | Failures: {fail_count}")

    results = {
        "interaction_counter": interaction_counter,
        "full_process_counter": full_process_counter,
        "cross_sections": CrossSections_list  # now holds all xsec + error
    }
    
    return results


    
#=======================================================================
#================================ LHE WRITE ============================
#=======================================================================

# Later (if I remember): add something that changes the pre-amble at the top of the LHE files per gen

def WriteLHE(events, file_name, s, PDF, cross_sections, lprup_by_index=None):
    # Get info from generation and put it into proper LHE format
    # cross_sections: list of tuples (xsec, error, max_weight, lprup)
    # lprup_by_index: list mapping InteractionIndex -> lprup (int)

    IDBMUP = 2212
    EBMUP = np.sqrt(s) / 2
    PDFGUP = 0
    PDFSUP = PDF.lhapdfID
    IDWTUP = 3
    NPRUP = len(cross_sections)

    with open(file_name, "w") as fh:
        fh.write("<LesHouchesEvents version=\"1.0\">\n")
        fh.write("<init>\n")
        fh.write(f"{IDBMUP} {IDBMUP} {EBMUP} {EBMUP} "
                 f"{PDFGUP} {PDFGUP} {PDFSUP} {PDFSUP} {IDWTUP} {NPRUP}\n")

        # Write cross sections with their lprup numbers
        for xsec, err, maxw, lprup in cross_sections:
            fh.write(f"{xsec} {err} {maxw} {lprup}\n")

        fh.write("</init>\n")

        # Write events 
        for event_data in events:
            (event, xa, xb, M, ID1, ID2, FullIDs, InteractionIndex,
             lprup, selected_process, chosen_colour_flow) = event_data
    
            outIDs = FullIDs[2:]
            NUP = 2 + len(outIDs)
            XWGTUP = 1.0
            PDFScale = 1000
            QED = 1 / 126.5
            alpha_s = 0.1
            StatusMother = -1
            StatusDaughter = 1
            MotherOne, MotherTwo = 1, 2
            IP = lprup

            fh.write("<event>\n")
            fh.write(f"{NUP} {IP} {XWGTUP} {PDFScale} {QED} {alpha_s}\n")

            # Incoming particles
            fh.write("{} {} 0 0 {} {} {: e} {: e} {: e} {: e} 0 0 9\n".format(
                ID1, StatusMother,
                chosen_colour_flow[0][0], chosen_colour_flow[0][1],
                event[0][1], event[0][2], event[0][3], event[0][0]))

            fh.write("{} {} 0 0 {} {} {: e} {: e} {: e} {: e} 0 0 9\n".format(
                ID2, StatusMother,
                chosen_colour_flow[1][0], chosen_colour_flow[1][1],
                event[1][1], event[1][2], event[1][3], event[1][0]))

            # Outgoing particles
            for i, outID in enumerate(outIDs, start=2):
                flow = chosen_colour_flow[i] if chosen_colour_flow and i < len(chosen_colour_flow) else (0, 0)
                p = event[i]
                fh.write("{} {} {} {} {} {} {: e} {: e} {: e} {: e} 0 0 9\n".format(
                    outID, StatusDaughter, MotherOne, MotherTwo,
                    flow[0], flow[1],
                    p[1], p[2], p[3], p[0]))

            # PDF comment line
            fh.write(f"#pdf {ID1} {ID2} {xa} {xb} {PDFScale} "
                     f"{PDF.xfxQ(ID1, xa, PDFScale)} {PDF.xfxQ(ID2, xb, PDFScale)}\n")
            fh.write("</event>\n")

        fh.write("</LesHouchesEvents>\n")



#=====================================================================
#============================== OUTPUT ===============================
#=====================================================================

pretty_names_2to2 = {
    "gg_gg":       "gg → gg",
    "gg_qqx":      "gg → qq̄",
    "qqx_gg":      "qq̄ → gg",
    "gq_gq":       "gq → gq",
    "gqx_gqx":     "gq̄ → gq̄",
    "qqp_qqp":     "qq' → qq'",
    "qqpx_qqpx":   "qq̄' → qq̄'",
    "qxqpx_qxqpx": "q̄q̄' → q̄q̄'",
    "qqx_qpqpx":   "qq̄ → q'q̄'",
    "qq_qq":       "qq → qq",
    "qxqx_qxqx":   "q̄q̄ → q̄q̄",
    "qqx_qqx":     "qq̄ → qq̄",
}


pretty_names_2to3 = {

    # (1) gg → gg
    "gg_ggg":   "gg → ggg",
    "gg_qqxg":  "gg → qq̄g",

    # (2) gg → qq̄
    "gg_qqx":   "gg → qq̄g",

    # (3) qq̄ → gg
    "qqx_ggg":      "qq̄ → ggg",
    "qqx_qqxg":     "qq̄ → qq̄g",        
    "qqx_qpqpxg":   "qq̄ → q′q̄′g",

    # (4) gq → gq
    "gq_gqg":       "gq → gqg",
    "gq_qqxq":      "gq → qq̄q",
    "gq_qpqpxq":    "gq → q′q̄′q",

    # (5) gq̄ → gq̄
    "gqx_gqxg":     "gq̄ → gq̄g",
    "gqx_qqxqx":    "gq̄ → qq̄q̄",
    "gqx_qpqpxqx":  "gq̄ → q′q̄′q̄",

    # (6) qq′ → qq′
    "qqp_qqpg":     "qq′ → qq′g",

    # (7) qq̄′ → qq̄′
    "qqpx_qqpxg":   "qq̄′ → qq̄′g",

    # (8) q̄q̄′ → q̄q̄′
    "qxqpx_qxqpxg": "q̄q̄′ → q̄q̄′g",

    # (9) qq̄ → q′q̄′
    "qqx_qpqpxg":   "qq̄ → q′q̄′g",

    # (10) qq → qq
    "qq_qqg":       "qq → qqg",

    # (11) q̄q̄ → q̄q̄
    "qxqx_qxqxg":   "q̄q̄ → q̄q̄g",

    # (12) qq̄ → qq̄
    "qqx_qqxg":     "qq̄ → qq̄g",
}


pretty_names_2to4 = {
    # (1) gg → gg
    "gg_gggg":     "gg → gggg",
    "gg_ggqqx":    "gg → ggqq̄",
    "gg_qqxqqx":   "gg → qq̄qq̄",
    "gg_qqxqpqpx": "gg → qq̄q'q̄'",

    # (2) gg → qq̄
    "gg_qqxqqx":   "gg → qq̄qq̄",
    "gg_qqxqpqpx": "gg → qq̄q'q̄'",
    "gg_ggqqx":    "gg → ggqq̄",

    # (3) qq̄ → gg
    "qqx_gggg":     "qq̄ → gggg",
    "qqx_ggqqx":    "qq̄ → ggqq̄",
    "qqx_ggqpqpx":  "qq̄ → ggq'q̄'",
    "qqx_qqxqqx":   "qq̄ → qq̄qq̄",
    "qqx_qqxqpqpx": "qq̄ → qq̄q'q̄'",
    "qqx_qpqpxqpqpx":"qq̄ → q'q̄'q'q̄'",
    "qqx_qpqpxoox":  "qq̄ → q'q̄' q'' q̄''",  

    # (4) gq → gq
    "gq_gqgg":      "gq → gqgg",
    "gq_gqqqx":     "gq → gqqq̄",
    "gq_gqqpqpx":   "gq → gqq'q̄'",

    # (5) gq̄ → gq̄
    "gqx_gqxgg":    "gq̄ → gq̄gg",
    "gqx_gqxqqx":   "gq̄ → gq̄qq̄",
    "gqx_gqxqpqpx": "gq̄ → gq̄q'q̄'",

    # (6) qq' → qq' 
    "qqp_qqpgg":    "qq' → qq'gg",
    "qqp_qqpqqx":   "qq' → qq'qq̄",
    "qqp_qqpqpqpx": "qq' → qq'q'q̄'",
    "qqp_qqpoox":   "qq' → qq'q''q̄''",  

    # (7) qq̄' → qq̄'
    "qqpx_qqpxgg":      "qq̄' → qq̄'gg",
    "qqpx_qqpxqqx":     "qq̄' → qq̄'qq̄",
    "qqpx_qqpxqpqpx":   "qq̄' → qq̄'q'q̄'",
    "qqpx_qqpxoox":     "qq̄' → qq̄'q''q̄''",  

    # (8) q̄q̄' → q̄q̄' 
    "qxqpx_qxqpxgg":      "q̄q̄' → q̄q̄'gg",
    "qxqpx_qxqpxqqx":     "q̄q̄' → q̄q̄'qq̄",
    "qxqpx_qxqpxqpqpx":   "q̄q̄' → q̄q̄'q'q̄'",
    "qxqpx_qxqpxoox":     "q̄q̄' → q̄q̄'q''q̄''", 

    # (9) qq̄ → q'q̄' 
    "qqx_qpqpxgg":    "qq̄ → q'q̄'gg",
    "qqx_qqxqpqpx":   "qq̄ → qq̄q'q̄'",
    "qqx_qpqpxqpqpx": "qq̄ → q'q̄'q'q̄'",
    "qqx_qpqpxoox":   "qq̄ → q'q̄'q''q̄''", 

    # (10) qq → qq 
    "qq_qqgg":      "qq → qqgg",
    "qq_qqqqx":     "qq → qqqq̄",
    "qq_qqqpqpx":   "qq → qqq'q̄'",

    # (11) q̄q̄ → q̄q̄ 
    "qxqx_qxqxgg":    "q̄q̄ → q̄q̄gg",
    "qxqx_qxqxqqx":   "q̄q̄ → q̄q̄qq̄",
    "qxqx_qxqxqpqpx": "q̄q̄ → q̄q̄q'q̄'",

    # (12) qq̄ → qq̄ 
    "qqx_qqxgg":      "qq̄ → qq̄gg",
    "qqx_qqxqqx":     "qq̄ → qq̄qq̄",
    "qqx_qqxqpqpx":   "qq̄ → qq̄q'q̄'",
}


pretty_names_2to5 = {

    # (1) gg → gg
    "gg_ggqqxg":        "gg → ggqq̄g",
    "gg_qqxqqxg":       "gg → qq̄qq̄g",
    "gg_qqxqpqpxg":     "gg → qq̄q′q̄′g",

    # (2) gg → qq̄
    "gg_ggqqxg":        "gg → ggqq̄g",
    "gg_qqxqqxg":       "gg → qq̄qq̄g",
    "gg_qqxqpqpxg":     "gg → qq̄q′q̄′g",

    # (3) qq̄ → gg 
    "qqx_ggggg":        "qq̄ → ggggg",
    "qqx_ggqqxg":       "qq̄ → ggqq̄g",
    "qqx_ggqpqpxg":     "qq̄ → ggq′q̄′g",
    "qqx_qqxqqxg":      "qq̄ → qq̄qq̄g",
    "qqx_qqxqpqpxg":    "qq̄ → qq̄q′q̄′g",
    "qqx_qpqpxqpqpxg":  "qq̄ → q′q̄′q′q̄′g",
    "qqx_qpqpxooxg":    "qq̄ → q′q̄′q″q̄″g",  

    # (4) gq → gq 
    "gq_gqggg":         "gq → gqggg",
    "gq_gqqqxg":        "gq → gqqq̄g",
    "gq_gqqpqpxg":      "gq → gqq′q̄′g",
    "gq_qqxqqxq":       "gq → qq̄qq̄q",
    "gq_qqxqpqpxq":     "gq → qq̄q′q̄′q",
    "gq_qpqpxqpqpxq":   "gq → q′q̄′q′q̄′q",
    "gq_qpqpxooxq":     "gq → q′q̄′q″q̄″q",

    # (5) gq̄ → gq̄
    "gqx_gqxggg":       "gq̄ → gq̄ggg",
    "gqx_gqxqqxg":      "gq̄ → gq̄qq̄g",
    "gqx_gqxqpqpxg":    "gq̄ → gq̄q′q̄′g",
    "gqx_qqxqqxqx":     "gq̄ → qq̄qq̄q̄",
    "gqx_qpqpxqqxqx":   "gq̄ → q′q̄′qq̄q̄",
    "gqx_qpqpxqpqpxqx": "gq̄ → q′q̄′q′q̄′q̄",
    "gqx_qpqpxooxqx":   "gq̄ → q′q̄′q″q̄″q̄",

    # (6) qq′ → qq′ 
    "qqp_qqpggg":       "qq′ → qq′ggg",
    "qqp_qqpqqxg":      "qq′ → qq′qq̄g",
    "qqp_qqpqpqpxg":    "qq′ → qq′q′q̄′g",
    "qqp_qqpooxg":      "qq′ → qq′q″q̄″g",

    # (7) qq̄′ → qq̄′ 
    "qqpx_qqpxggg":     "qq̄′ → qq̄′ggg",
    "qqpx_qqpxqqxg":    "qq̄′ → qq̄′qq̄g",
    "qqpx_qqpxqpqpxg":  "qq̄′ → qq̄′q′q̄′g",
    "qqpx_qqpxooxg":    "qq̄′ → qq̄′q″q̄″g",

    # (8) q̄q̄′ → q̄q̄′ 
    "qxqpx_qxqpxggg":       "q̄q̄′ → q̄q̄′ggg",
    "qxqpx_qxqpxqqxg":      "q̄q̄′ → q̄q̄′qq̄g",
    "qxqpx_qxqpxqpqpxg":    "q̄q̄′ → q̄q̄′q′q̄′g",
    "qxqpx_qxqpxooxg":      "q̄q̄′ → q̄q̄′q″q̄″g",

    # (9) qq̄ → q′q̄′
    "qqx_qpqpxggg":        "qq̄ → q′q̄′ggg",
    "qqx_qqxqpqpxg":       "qq̄ → qq̄q′q̄′g",
    "qqx_qpqpxqpqpxg":     "qq̄ → q′q̄′q′q̄′g",
    "qqx_qpqpxooxg":       "qq̄ → q′q̄′q″q̄″g",

    # (10) qq → qq 
    "qq_qqggg":            "qq → qqggg",
    "qq_qqqqxg":           "qq → qqqq̄g",
    "qq_qqqpqpxg":         "qq → qqq′q̄′g",

    # (11) q̄q̄ → q̄q̄ 
    "qxqx_qxqxggg":        "q̄q̄ → q̄q̄ggg",
    "qxqx_qxqxqqxg":       "q̄q̄ → q̄q̄qq̄g",
    "qxqx_qxqxqpqpxg":     "q̄q̄ → q̄q̄q′q̄′g",

    # (12) qq̄ → qq̄ 
    "qqx_qqxggg":          "qq̄ → qq̄ggg",
    "qqx_qqxqqxg":         "qq̄ → qq̄qq̄g",
    "qqx_qqxqpqpxg":       "qq̄ → qq̄q′q̄′g",
}

pretty_names = {}

# I don't think this is working all the time but it's not imperative for the code to work, just quality of life thing...
def format_summary(interaction_counter, full_process_counter, total_events):
    # Process info output
    lines = []

    for proc_name in interaction_counter:
        total = interaction_counter[proc_name]
        frac = 100.0 * total / total_events
        lines.append(f"{pretty_names.get(proc_name, proc_name)}: {total} events ({frac:.1f}%)")

        grouped_events = {}
        for (name, init_ids, sel_proc, FullIDs), count in full_process_counter.items():
            if name != proc_name:
                continue

            # First two entries = incoming, rest = outgoing
            outgoing_ids = tuple(FullIDs[2:])

            key = (tuple(init_ids), sel_proc, outgoing_ids)
            grouped_events[key] = grouped_events.get(key, 0) + count

        for (init_ids, sel_proc, outgoing_ids), count in grouped_events.items():
            subfrac = 100.0 * count / total
            init_txt = ",".join(map(str, init_ids))
            out_txt = ",".join(map(str, outgoing_ids))

            # Determine pretty-name dictionary based on number of outgoing particles
            n_out = len(outgoing_ids)
            if n_out == 2:
                pretty_dict = pretty_names_2to2
            elif n_out == 3:
                pretty_dict = pretty_names_2to3
            elif n_out == 4:
                pretty_dict = pretty_names_2to4
            elif n_out == 5:
                pretty_dict = pretty_names_2to5
            else:
                pretty_dict = {}  # safety fallback

            if sel_proc:
                if isinstance(sel_proc, tuple):
                    sel_key = sel_proc[0]
                else:
                    sel_key = sel_proc
                sel_txt = pretty_dict.get(sel_key, sel_key)
                mid_txt = f"{sel_txt:<12} | "
            else:
                mid_txt = ""

            lines.append(
                f"    Initial IDs = {init_txt} | "
                f"{mid_txt}"
                f"Outgoing IDs = {out_txt} | "
                f"Events: {count} ({subfrac:.1f}%)"
            )

        lines.append("")

    return "\n".join(lines)