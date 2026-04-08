#!/usr/bin/env python
# coding: utf-8


import numpy as np 
import random as ran 
import matplotlib.pyplot as plt 
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict


import re
import glob 
import os


#======================================================================
#========================= TOLERANCE CHECKER ==========================
#======================================================================

def apply_tolerance(P, tol=1e-14):
    P = np.array(P)  #Must be a numpy array
    P[np.abs(P) < tol] = 0
    return P


#======================================================================
#========================= MASS FUNCTIONS =============================
#======================================================================

def m(p):
    epsilon = 1e-10 #Tolerance for numerical errors (small masses will be forced to 0)
    m_squared = p[0]**2 - np.dot(p[1:], p[1:]) #4-vector dot product in Minkowski space
    if -epsilon < abs(m_squared) < epsilon: m_squared = 0
    elif m_squared < 0: print("Warning: Negative mass")
    return np.sqrt(m_squared)


def calculate_masses(four_momenta_list):
    particle_indices = []
    masses = []
    for event_index, event in enumerate(four_momenta_list):
        for particle_index, particle in enumerate(event):
            mass = m(particle)  # Calculate mass for each particle
            particle_indices.append(f"Event {event_index + 1}, Particle {particle_index + 1}")
            masses.append(mass)
    return masses 


#======================================================================
#========================= BOOST FUNCTIONS ============================
#======================================================================

def boost(gamma_rel, beta_rel, p):
    b = np.sqrt(np.dot(beta_rel, beta_rel))
    b_hat = beta_rel / b
    Ep = p[0]
    pp = p[1:]
    E = gamma_rel*(Ep + np.dot(pp , b_hat )*b)
    P = pp + (gamma_rel -1.0) * np.dot(pp , b_hat )*b_hat + gamma_rel * Ep * beta_rel
    return np.concatenate ((E ,P) , axis = None )


def boost_to_lab(event, gamma_rel, beta_rel, debug=False):
    boosted_event = []
    for i, p in enumerate(event):
        boosted_p = boost(gamma_rel, beta_rel, p)  # Apply the boost
        mass = m(boosted_p)  # Calculate mass after boost
        
        if debug:
            print(f"Particle {i+1} lab vector: {boosted_p}")
            print(f"Particle {i+1} mass after boost: {mass}")
            print()
            
        if mass > 1e-10:  # Ensure the particle is effectively massless
            print(f"Warning: Particle {i+1} has non-zero mass after boost!")
        boosted_event.append(boosted_p)
    return boosted_event


def boost_all(momenta_list, gamma_rel, beta_rel, debug=False):
    boosted_momenta_list = []
    for event in momenta_list:
        boosted_event = boost_to_lab(event, gamma_rel, beta_rel, debug=debug)
        boosted_momenta_list.append(boosted_event)
    return boosted_momenta_list


def boost_to_com(event, debug=False):
    # Calculate total energy and total momentum
    total_energy = sum(p[0] for p in event)
    total_momentum = np.sum([p[1:] for p in event], axis=0)
    
    # Calculate the invariant mass M
    M = np.sqrt(total_energy**2 - np.dot(total_momentum, total_momentum))

    # Calculate the boost parameters to the CoM frame
    beta = total_momentum / total_energy
    beta_mag = np.linalg.norm(beta)
    gamma = 1.0 / np.sqrt(1.0 - beta_mag**2)

    if debug:
        print(f"Total energy: {total_energy:.6f}")
        print(f"Total momentum: {total_momentum}")
        print(f"Invariant mass M: {M:.6f}")
        print(f"Beta vector: {beta}")
        print(f"Gamma factor: {gamma:.6f}\n")

    # Boost each particle to the CoM frame
    boosted_event = [boost(gamma, -beta, p) for p in event]

    return M, boosted_event



#======================================================================
#========================= GENERAL PLOTTING ===========================
#======================================================================

def plot_four_vectors_2d(p1, p2, p3, p4, label_fontsize, legend_fontsize, tick_fontsize, ax=None):
    #Extract the spatial components (px, py, pz) of each 4-vector
    p1_spatial = np.array(p1[1:])
    p2_spatial = np.array(p2[1:])
    p3_spatial = np.array(p3[1:])
    p4_spatial = np.array(p4[1:])

    #Create a new figure and axis if no axes are provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        if plot_type == "3D":
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
    else:
        fig = None

    #Plot each vector's xy components with fixed scaling
    ax.quiver(0, 0, p1_spatial[0], p1_spatial[1], color='r', angles='xy', scale_units='xy', scale=1, label=r"$d_1$")
    ax.quiver(0, 0, p2_spatial[0], p2_spatial[1], color='k', angles='xy', scale_units='xy', scale=1, label=r"$d_2$")
    ax.quiver(0, 0, p3_spatial[0], p3_spatial[1], color='b', angles='xy', scale_units='xy', scale=1, label=r"$d_3$")
    ax.quiver(0, 0, p4_spatial[0], p4_spatial[1], color='y', angles='xy', scale_units='xy', scale=1, label=r"$d_4$")

    #Set axis labels
    ax.set_xlabel(r"$P_x$ (TeV)", fontsize=label_fontsize)
    ax.set_ylabel(r"$P_y$ (TeV)", fontsize=label_fontsize)

    #Set axis limits based on the maximum vector component, adding some margin
    max_val = np.max(np.abs([p1_spatial, p2_spatial, p3_spatial, p4_spatial])) * 1.1
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_aspect('equal')

    #Picture formatting
    ax.legend(fontsize=legend_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    ax.grid(True)

    #Show the plot only if a new figure was created
    if fig is not None:
        plt.tight_layout()
        plt.show()
    
    return fig if fig is not None else ax  #Return the figure if new, else return the existing axes


def plot_four_vectors_3d(p1, p2, p3, p4, label_fontsize, legend_fontsize, tick_fontsize, ax=None):
    #Extract the spatial components (px, py, pz) of each 4-vector
    p1_spatial = np.array(p1[1:])
    p2_spatial = np.array(p2[1:])
    p3_spatial = np.array(p3[1:])
    p4_spatial = np.array(p4[1:])

    #Create a new figure and 3D axis if none are provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = None  #Track if a new figure was created

    #Plot each 4-vector's spatial components
    ax.quiver(0, 0, 0, p1_spatial[0], p1_spatial[1], p1_spatial[2], color='r', label=r"$d_1$")
    ax.quiver(0, 0, 0, p2_spatial[0], p2_spatial[1], p2_spatial[2], color='0', label=r"$d_2$")
    ax.quiver(0, 0, 0, p3_spatial[0], p3_spatial[1], p3_spatial[2], color='b', label=r"$d_3$")
    ax.quiver(0, 0, 0, p4_spatial[0], p4_spatial[1], p4_spatial[2], color='y', label=r"$d_4$")

    #Set axis labels
    ax.set_xlabel(r"$P_x$ (TeV)", fontsize=label_fontsize)
    ax.set_ylabel(r"$P_y$ (TeV)", fontsize=label_fontsize)
    #ax.set_zlabel('Pz (TeV)', fontsize=label_fontsize)

    #Set axis limits based on the maximum vector component, adding some margin
    max_val = np.max(np.abs([p1_spatial, p2_spatial, p3_spatial, p4_spatial])) * 1.1
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    ax.legend(fontsize=legend_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    ax.grid(True)

    #Show the plot only if we created a new figure (to avoid double plotting in case of subplots)
    if fig is not None:
        plt.tight_layout()
        plt.show()

    return ax


#======================================================================
#========================= ROTATION FUNCTIONS =========================
#======================================================================

def rotate_xy(p, theta):
    #Rotation matrix in the xy-plane
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    
    #Extract the xy-components of the momentum
    p_x, p_y = p[1], p[2]
    
    #Apply the rotation to the (x, y) components
    p_xy_rotated = np.dot(rotation_matrix, np.array([p_x, p_y]))
    
    #Return the rotated 4-vector
    return [p[0], p_xy_rotated[0], p_xy_rotated[1], p[3]]


def Euler_rotation(alpha, beta, gamma):
    #Rotation around z-axis (alpha)
    Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                   [np.sin(alpha), np.cos(alpha),  0],
                   [0,            0,             1]])
    
    #Rotation around y-axis (beta)
    Ry = np.array([[np.cos(beta),  0, np.sin(beta)],
                   [0,            1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    
    #Rotation around x-axis (gamma)
    Rx = np.array([[1, 0,             0],
                   [0, np.cos(gamma), -np.sin(gamma)],
                   [0, np.sin(gamma), np.cos(gamma)]])
    
    #Combine rotations with the dot product
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    return R


def rotate_3d(p, alpha, beta, gamma):
    #Extract the spatial components (px, py, pz)
    p_spatial = np.array([p[1], p[2], p[3]])

    #Generate the 3D rotation matrix
    R = Euler_rotation(alpha, beta, gamma)
    
    #Apply the rotation to the spatial components
    p_spatial_rotated = np.dot(R, p_spatial)

    #Return the rotated 4-vector (energy stays the same)
    return [p[0], p_spatial_rotated[0], p_spatial_rotated[1], p_spatial_rotated[2]]


#===========================================================================
#========================= MASSLESS EVENT GENERATOR ========================
#===========================================================================

from scipy.special import gamma as gamma_func

def rambo(n, w, P, TwoD, ThreeD):
    #Generate N massless momenta in infinite phase space
    Q = np.zeros((4, n))  #4xN array for 4-momentum components
    TwoPi = 2 * np.pi
    
    #Weight function that is not specicially used here:
    V = (np.pi / 2) ** (n - 1) * (w ** (2 * n - 4) / (gamma_func(n) * gamma_func(n - 1))) 
    
    for I in range(n):
        #This loop calculates equations from eq 3.1 
        if TwoD:
            C = 0
        if ThreeD:
            C = 2 * np.random.uniform() - 1    #This is C_i
            
        S = np.sqrt(1 - C**2)              #This is for q_i's
        phi = TwoPi * np.random.uniform()  #This is phi_i
        
        Q[3, I] = np.log(np.random.uniform() * np.random.uniform())    #q_i^0                                    
        Q[2, I] = Q[3, I] * C                                          #q_i^z
        Q[1, I] = Q[3, I] * S * np.cos(phi)                            #q_i^x
        Q[0, I] = Q[3, I] * S * np.sin(phi)                            #q_i^y
        
    #Calculate parameters of the conformal transformation from the paper:
    M = np.zeros(4)
    
    for K in range(4):
        M[K] = np.sum(Q[K, :n])  #Sum over the first N elements for each component

    Big_M = np.sqrt(M[3]**2 - M[2]**2 - M[1]**2 - M[0]**2)
    #print("Mass of system?:", Big_M)
    
    b = np.zeros(3)
    for K in range(3):
        b[K] = - M[K] / Big_M

    G = M[3] / Big_M
    a = 1 / (1 + G)
    x = w / Big_M

    P_array = []
    #Transform the Q's conformally into the P's
    for I in range(n):
        b_dot_q = b[0] * Q[0, I] + b[1] * Q[1, I] + b[2] * Q[2, I]
        for K in range(3):
            P[K, I] = x * (Q[K, I] + (b[K]*Q[3, I]) + a*b_dot_q*b[K])
        P[3, I] = x * (G*Q[3, I] + b_dot_q)
        
        #Append the 4-momentum as [E, Px, Py, Pz]
        P_array.append([P[3, I], P[1, I], P[0, I], P[2, I]])

    p1_com = P_array[0]
    p2_com = P_array[1]
    p3_com = P_array[2] 
    p4_com = P_array[3]

    return p1_com, p2_com, p3_com, p4_com #Spherical event four vectors


def planar_or_spherical():
    n = 4                  #Number of particles. This is always 4 in our case 
    w = 4                  #Total center-of-mass energy
    P = np.zeros((4, n))   #Array to store the 4-momentum of each particle. 

    #Binomial probability
    p = 0.5
    planar_or_spherical_choice = np.random.choice([1, 2], p=[p, 1 - p])
    
    if planar_or_spherical_choice == 1:
        #Call one of the planar events
        return rambo(w, n, P, TwoD = True, ThreeD = False)
        
    elif planar_or_spherical_choice == 2:
        #Call the arbitrary spherical event
        return rambo(w, n, P, TwoD = False, ThreeD = True)


def gen_CoM_4_mom(N, Normal= False, SplitSym= False, DR = False, w=None, n=None, P=None):
    momenta_list = []
    #planar_tally = 0 
    #spherical_tally = 0
    if Normal:
        for _ in range(N):
            p1cms, p2cms, p3cms, p4cms = rambo(w, n, P, TwoD = False, ThreeD = True)
            momenta_list.append([p1cms, p2cms, p3cms, p4cms])
    elif SplitSym:    
        for _ in range(N):
            p1cms, p2cms, p3cms, p4cms = planar_or_spherical() # Generate a single event
            momenta_list.append([p1cms, p2cms, p3cms, p4cms]) # Append the full 4-momenta of all particles for this event
    
            # Track events 
            #if planar_or_spherical_choice == 1: 
            #    planar_tally += 1
    
            #if planar_or_spherical_choice == 2: 
            #    spherical_tally += 1

    elif DR:
        for _ in range(N):
            p1cms, p2cms, p3cms, p4cms = rambo(w, n, P, TwoD = True, ThreeD = False)
            momenta_list.append([p1cms, p2cms, p3cms, p4cms])

    else:
        return "Error: No valid flag selected."
    
    return momenta_list


#===============================================================
#========================= LHE ANALYSIS ========================
#===============================================================

def diff_momentum(four_momentum_list, mode="all"):
    three_mom_all = []
    energy_list   = []
    momentum_list = []
    pt_list       = []
    eta_list      = []     # per event
    phi_list      = []     # per event
    theta_list    = []     # per event
    delta_eta_list = []    # flat list: leading_eta - subleading_eta (one value per event with >=2 particles)
    delta_theta_list = []  # flat list: leading_theta - subleading_theta (one value per event with >=2 particles)
    delta_phi_list = []
    px_list       = []
    py_list       = []
    pz_list       = []

    for event in four_momentum_list:
        # 3-vectors for event-shape variables
        event_three_mom = [[px, py, pz] for (_, px, py, pz) in event]
        three_mom_all.append(event_three_mom)

        # pT ranking for the event
        pts = [np.sqrt(px**2 + py**2) for (_, px, py, pz) in event]
        sort_idx = np.argsort(pts)[::-1]  # indices of particles sorted by pT descending

        # choose particles according to mode (keeps compatibility with your previous code)
        if mode == "All":
            selected_particles = event
        elif mode == "Leading" and len(sort_idx) >= 1:
            selected_particles = [event[sort_idx[0]]]
        elif mode == "Subleading" and len(sort_idx) >= 2:
            selected_particles = [event[sort_idx[1]]]
        elif mode == "Tertiary" and len(sort_idx) >= 3:
            selected_particles = [event[sort_idx[2]]]
        elif mode == "Last" and len(sort_idx) >= 4:
            selected_particles = [event[sort_idx[3]]]
        else:
            selected_particles = []

        # per-event containers (for selected_particles)
        event_E     = []
        event_p     = []
        event_pt    = []
        event_px    = []
        event_py    = []
        event_pz    = []
        event_eta   = []
        event_phi   = []
        event_theta = []

        for (E, px, py, pz) in selected_particles:
            p_mag = np.sqrt(px**2 + py**2 + pz**2)
            pt    = np.sqrt(px**2 + py**2)
            theta = np.arccos(pz / p_mag) if p_mag > 0 else np.nan
            eta   = -np.log(np.tan(theta / 2.0)) if theta > 0 else np.nan
            phi   = np.arctan2(py, px)

            event_E.append(E)
            event_p.append(p_mag)
            event_pt.append(pt)
            event_px.append(px)
            event_py.append(py)
            event_pz.append(pz)
            event_eta.append(eta)
            event_phi.append(phi)
            event_theta.append(theta)

        # compute leading - subleading differences from the event (based on pT)
        if len(event) >= 2:
            i0, i1 = sort_idx[0], sort_idx[1]
            E0, px0, py0, pz0 = event[i0]
            E1, px1, py1, pz1 = event[i1]

            p0_mag = np.sqrt(px0**2 + py0**2 + pz0**2)
            p1_mag = np.sqrt(px1**2 + py1**2 + pz1**2)

            theta0 = np.arccos(pz0 / p0_mag) if p0_mag > 0 else np.nan
            theta1 = np.arccos(pz1 / p1_mag) if p1_mag > 0 else np.nan
            eta0   = -np.log(np.tan(theta0 / 2.0)) if theta0 > 0 else np.nan
            eta1   = -np.log(np.tan(theta1 / 2.0)) if theta1 > 0 else np.nan
            phi0   = np.arctan2(py0, px0)
            phi1   = np.arctan2(py1, px1)

            delta_eta_list.append(eta0 - eta1)
            delta_theta_list.append(theta0 - theta1)
            
            # compute wrapped phi difference
            #dphi = phi0 - phi1
            #dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
            delta_phi_list.append(phi0-phi1)   

        # store per-event lists (for selected_particles)
        energy_list.append(event_E)
        momentum_list.append(event_p)
        pt_list.append(event_pt)
        px_list.append(event_px)
        py_list.append(event_py)
        pz_list.append(event_pz)
        eta_list.append(event_eta)
        phi_list.append(event_phi)
        theta_list.append(event_theta)

    return (
        three_mom_all,
        energy_list,
        momentum_list,
        pt_list,
        eta_list,
        phi_list,
        px_list,
        py_list,
        pz_list,
        theta_list,
        delta_eta_list,
        delta_theta_list,
        delta_phi_list
    )


# helper for mean, RMS, and all that for plots
def annotate_stats(ax, data, bin_edges=None, include_flow=False):
    """Annotate plot with entries, mean, RMS, and optional flow info."""
    data = np.asarray(data)
    n_entries = data.size
    mean = np.nanmean(data) if n_entries else 0.0
    rms = np.nanstd(data) if n_entries else 0.0

    text = f"Entries = {n_entries}\nMean = {mean:.2f}\nRMS = {rms:.2f}"
    if include_flow and bin_edges is not None:
        under = np.sum(data < bin_edges[0])
        over = np.sum(data > bin_edges[-1])
        text += f"\nUnderflow = {under}\nOverflow = {over}"

    ax.text(
        0.98, 0.95, text,
        transform=ax.transAxes,
        fontsize=12, verticalalignment='top', horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
    )


def plot_kinematics(
    energy, momentum, pt, eta, phi, px, py, pz,
    theta=None,
    output_file_prefix=None, file_format='png'
):
    label_fontsize = 16

    def format_bin_width(val, unit):
        """Format bin width nicely depending on scale."""
        if val < 1:
            return f"{val:.2f} {unit}"
        elif val < 10:
            return f"{val:.1f} {unit}"
        elif val < 100:
            return f"{round(val)} {unit}"
        else:
            return f"{round(val, -1)} {unit}"

    # =========================
    # 1) Energy, |p|, pT
    # =========================
    all_eppt = energy + momentum + pt
    num_bins_eppt = 80
    edges = np.linspace(0, max(all_eppt), num_bins_eppt + 1)
    bin_w = edges[1] - edges[0]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for ax, data, lbl in zip(
        axes, [energy, momentum, pt],
        ["Energy [GeV]", r"$|\vec p|$ [GeV]", r"$p_T$ [GeV]"]
    ):
        counts, bins_, _ = ax.hist(data, bins=edges, color='red', alpha=0.7,
                                   edgecolor='gray', linewidth=1.2)
        ax.set_xlabel(lbl, fontsize=label_fontsize)
        ax.set_ylabel(f"Events / {format_bin_width(bin_w, 'GeV')}", fontsize=label_fontsize)
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.7)
        annotate_stats(ax, data, bins_)
    plt.tight_layout()
    if output_file_prefix:
        plt.savefig(f"{output_file_prefix}_eppt.{file_format}", bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # =========================
    # 2) px, py, pz
    # =========================
    all_xyz = px + py + pz
    num_bins_xyz = 50
    edges = np.linspace(min(all_xyz), max(all_xyz), num_bins_xyz + 1)
    bin_w = edges[1] - edges[0]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for ax, data, lbl in zip(
        axes, [px, py, pz],
        [r"$p_x$ [GeV]", r"$p_y$ [GeV]", r"$p_z$ [GeV]"]
    ):
        counts, bins_, _ = ax.hist(data, bins=edges, color='blue', alpha=0.7,
                                   edgecolor='gray', linewidth=1.2)
        ax.set_xlabel(lbl, fontsize=label_fontsize)
        ax.set_ylabel(f"Events / {format_bin_width(bin_w, 'GeV')}", fontsize=label_fontsize)
        ax.grid(True, linestyle='--', alpha=0.7)
        annotate_stats(ax, data, bins_)
    plt.tight_layout()
    if output_file_prefix:
        plt.savefig(f"{output_file_prefix}_pxpypz.{file_format}", bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # =========================
    # 3) Rapidity (η)
    # =========================
    num_bins_eta = 40
    edges_eta = np.linspace(-5, 5, num_bins_eta + 1)
    bin_w_eta = edges_eta[1] - edges_eta[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # linear
    c, bins_, _ = axes[0].hist(eta, bins=edges_eta, color='green', alpha=0.7,
                               edgecolor='gray', linewidth=1.2)
    axes[0].set_xlabel(r'$\eta$', fontsize=label_fontsize)
    axes[0].set_ylabel(f"Events / {format_bin_width(bin_w_eta, '')}", fontsize=label_fontsize)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    annotate_stats(axes[0], eta, bins_, include_flow=True)
    # log
    c2, _, _ = axes[1].hist(eta, bins=edges_eta, color='green', alpha=0.7,
                            edgecolor='gray', linewidth=1.2)
    axes[1].set_xlabel(r'$\eta$', fontsize=label_fontsize)
    axes[1].set_yscale("log")
    axes[1].grid(True, linestyle='--', alpha=0.7)
    annotate_stats(axes[1], eta, edges_eta, include_flow=True)
    plt.tight_layout()
    if output_file_prefix:
        plt.savefig(f"{output_file_prefix}_eta.{file_format}", bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # =========================
    # 4) θ and φ
    # =========================
    num_bins_theta = 20
    num_bins_phi   = 20

    theta_edges = np.linspace(0, np.pi, num_bins_theta + 1)
    phi_edges   = np.linspace(-np.pi, np.pi, num_bins_phi + 1)

    bin_w_theta = theta_edges[1] - theta_edges[0]
    bin_w_phi   = phi_edges[1] - phi_edges[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if theta is not None and len(theta) > 0:
        counts, edges, _ = axes[0].hist(
            theta, bins=theta_edges,
            color='purple', alpha=0.7, edgecolor='gray', linewidth=1.2
        )
        axes[0].set_xlabel(r'$\theta$ [rad]', fontsize=label_fontsize)
        axes[0].set_ylabel(f"Events / {format_bin_width(bin_w_theta, 'rad')}", fontsize=label_fontsize)
        axes[0].grid(True, linestyle='--', alpha=0.7)
        annotate_stats(axes[0], theta, edges, include_flow=True)
    else:
        axes[0].text(0.5, 0.5, "No θ", ha='center', va='center')

    if phi is not None and len(phi) > 0:
        counts_phi, edges_phi, _ = axes[1].hist(
            phi, bins=phi_edges,
            color='orange', alpha=0.7, edgecolor='gray', linewidth=1.2
        )
        axes[1].set_xlabel(r'$\phi$ [rad]', fontsize=label_fontsize)
        axes[1].set_ylabel(f"Events / {format_bin_width(bin_w_phi, 'rad')}", fontsize=label_fontsize)
        axes[1].grid(True, linestyle='--', alpha=0.7)
        annotate_stats(axes[1], phi, edges_phi, include_flow=True)
    else:
        axes[1].text(0.5, 0.5, "No φ", ha='center', va='center')

    plt.tight_layout()
    if output_file_prefix:
        plt.savefig(f"{output_file_prefix}_theta_phi.{file_format}", bbox_inches="tight")
    plt.close(fig)
    # plt.show()
    
    
def plot_kinematics_overlay_full(
    data_by_mode,  # dict: {mode: {"energy":[], "momentum":[], "pt":[], "px":[], "py":[], "pz":[], "eta":[], "theta":[], "phi":[]}}
    output_file_prefix=None,
    file_format='png'
):
    import numpy as np
    import matplotlib.pyplot as plt

    styles = [
        dict(color="black",  linewidth=0.75, linestyle="-"),
        dict(color="orange", linewidth=3.5, linestyle="-"),
        dict(color="purple", linewidth=2.5, linestyle="--"),
        dict(color="blue",   linewidth=1, linestyle=":"),
        dict(color="gold",   linewidth=1, linestyle="-."),
    ]
    label_fontsize = 16

    def nice_bin_label(width):
        """Round bin width to a nice display value (1, 2, 5 × 10^n)."""
        if width <= 0:
            return width
        exp = np.floor(np.log10(width))
        frac = width / 10**exp
        if frac < 1.5:
            nice = 1
        elif frac < 3.5:
            nice = 2
        elif frac < 7.5:
            nice = 5
        else:
            nice = 10
        return nice * 10**exp

    def draw_stats_boxes(ax, data_dict, key, bin_edges=None, include_flow=False):
        """Draw stacked stats boxes per mode for given axis and key, with bold centered mode label."""
        for i, (mode, vals) in enumerate(data_dict.items()):
            arr = flatten_vals(vals.get(key, []))
            n_entries = arr.size
            mean = np.nanmean(arr) if n_entries else 0.0
            rms = np.nanstd(arr) if n_entries else 0.0

            mode_text = f"$\\bf{{{mode}}}$"
            stats_text = f"Entries = {n_entries}\nMean = {mean:.2f}\nRMS = {rms:.2f}"
            if include_flow and bin_edges is not None:
                under = np.sum(arr < bin_edges[0])
                over = np.sum(arr > bin_edges[-1])
                stats_text += f"\nUnderflow = {under}\nOverflow = {over}"

            full_text = f"{mode_text}\n{stats_text}"

            n_modes = len(data_dict)
            y = 0.95 - i * 0.9 / n_modes

            ax.text(
                1.02, y, full_text,
                transform=ax.transAxes,
                va='top', ha='left',
                fontsize=10,
                linespacing=1.2,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.85)
            )

    # ========== 1) Energy / |p| / pT ==========
    num_bins = 80
    # Helper: ensure values are flattened to 1D arrays before passing to matplotlib
    def flatten_vals(vals):
        """Flatten a possibly nested list/array into a 1D numpy array."""
        if vals is None:
            return np.array([])
        out = []
        for v in vals:
            if isinstance(v, (list, tuple, np.ndarray)):
                out.extend(list(v))
            else:
                out.append(v)
        return np.array(out)

    all_vals = []
    for mdat in data_by_mode.values():
        all_vals.extend(flatten_vals(mdat.get("energy", [])).tolist())
        all_vals.extend(flatten_vals(mdat.get("momentum", [])).tolist())
        all_vals.extend(flatten_vals(mdat.get("pt", [])).tolist())
    edges = np.linspace(0, max(all_vals), num_bins + 1)
    bin_w = nice_bin_label(edges[1] - edges[0])

    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
    labels = ["Energy [GeV]", r"$|\vec p|$ [GeV]", r"$p_T$ [GeV]"]
    handles, labels_leg = [], []

    for ax, key, lbl in zip(axes, ["energy", "momentum", "pt"], labels):
        for i, (mode, vals) in enumerate(data_by_mode.items()):
            style = styles[i % len(styles)]
            arr = flatten_vals(vals.get(key, []))
            print(f"{mode}: {len(arr)} entries for {key}")  
            h = ax.hist(arr, bins=edges, histtype="step", label=mode, **style)
            if ax is axes[0]:
                handles.append(h[2][0])
                labels_leg.append(mode)
        draw_stats_boxes(ax, data_by_mode, key)
        ax.set_xlabel(lbl, fontsize=label_fontsize)
        ax.set_yscale("log")
        ax.grid(True, ls="--", alpha=0.7)

    fig.text(0.07, 0.5, f"Events / {bin_w:.0f} GeV", va='center', rotation='vertical', fontsize=label_fontsize)
    fig.legend(handles, labels_leg, loc="upper center", ncol=len(labels_leg), fontsize=26, frameon=False)
    plt.subplots_adjust(top=0.85, wspace=0.35)
    if output_file_prefix:
        plt.savefig(f"{output_file_prefix}_eppt_overlay.{file_format}", bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # ========== 2) px / py / pz ==========
    num_bins_xyz = 50
    all_xyz = []
    for mdat in data_by_mode.values():
        all_xyz.extend(mdat["px"] + mdat["py"] + mdat["pz"])
    edges_xyz = np.linspace(min(all_xyz), max(all_xyz), num_bins_xyz + 1)
    bin_w_xyz = nice_bin_label(edges_xyz[1] - edges_xyz[0])

    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
    xyz_labels = [r"$p_x$ [GeV]", r"$p_y$ [GeV]", r"$p_z$ [GeV]"]
    for ax, key, lbl in zip(axes, ["px", "py", "pz"], xyz_labels):
        for i, (mode, vals) in enumerate(data_by_mode.items()):
            style = styles[i % len(styles)]
            arr = flatten_vals(vals.get(key, []))
            ax.hist(arr, bins=edges_xyz, histtype="step", **style)
        draw_stats_boxes(ax, data_by_mode, key)
        ax.set_xlabel(lbl, fontsize=label_fontsize)
        ax.grid(True, ls='--', alpha=0.7)

    fig.text(0.07, 0.5, f"Events / {bin_w_xyz:.0f} GeV", va='center', rotation='vertical', fontsize=label_fontsize)
    plt.subplots_adjust(top=0.85, wspace=0.35)
    if output_file_prefix:
        plt.savefig(f"{output_file_prefix}_pxpypz_overlay.{file_format}", bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # ========== 3) η ==========
    num_bins_eta = 40
    edges_eta = np.linspace(-5, 5, num_bins_eta + 1)
    bin_w_eta = nice_bin_label(edges_eta[1] - edges_eta[0])

    fig, axes = plt.subplots(1, 2, figsize=(18,8), sharey=False)

    for ax, scale in zip(axes[:2], ["linear", "log"]):
        for i, (mode, vals) in enumerate(data_by_mode.items()):
            style = styles[i % len(styles)]
            arr = flatten_vals(vals.get("eta", []))
            ax.hist(arr, bins=edges_eta, histtype="step", **style)
        draw_stats_boxes(ax, data_by_mode, "eta", bin_edges=edges_eta, include_flow=True)
        ax.set_xlabel(r"$\eta$", fontsize=label_fontsize)
        if scale=="log": ax.set_yscale("log")
        ax.grid(True, ls='--', alpha=0.7)

    fig.text(0.06, 0.5, f"Events / {bin_w_eta:.1f}", va='center', rotation='vertical', fontsize=label_fontsize)
    plt.subplots_adjust(top=0.85, wspace=0.35)
    if output_file_prefix:
        plt.savefig(f"{output_file_prefix}_eta_overlay.{file_format}", bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    # ========== 4) θ / φ / cosθ ==========
    num_bins_theta = 20
    num_bins_phi = 20
    theta_edges = np.linspace(0, np.pi, num_bins_theta + 1)
    cos_edges   = np.linspace(-1, 1, num_bins_theta + 1)
    phi_edges   = np.linspace(-np.pi, np.pi, num_bins_phi + 1)

    bin_w_theta = nice_bin_label(theta_edges[1] - theta_edges[0])
    bin_w_cos   = nice_bin_label(cos_edges[1] - cos_edges[0])
    bin_w_phi   = nice_bin_label(phi_edges[1] - phi_edges[0])

    # Create a 2x2 grid, with phi spanning the bottom row
    fig = plt.figure(figsize=(20, 19))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])  # bottom row a bit taller
    ax_theta = fig.add_subplot(gs[0, 0])
    ax_costh = fig.add_subplot(gs[0, 1])
    ax_phi   = fig.add_subplot(gs[1, :])  # spans both columns

    # θ distribution
    for i, (mode, vals) in enumerate(data_by_mode.items()):
        style = styles[i % len(styles)]
        ax_theta.hist(flatten_vals(vals.get("theta", [])), bins=theta_edges, histtype="step", **style)
    draw_stats_boxes(ax_theta, data_by_mode, "theta", bin_edges=theta_edges, include_flow=True)
    ax_theta.set_xlabel(r"$\theta$ [rad]", fontsize=label_fontsize)
    ax_theta.set_ylabel(f"Events / {bin_w_theta:.2f} rad", fontsize=label_fontsize)
    ax_theta.grid(True, ls='--', alpha=0.7)

    # cosθ distribution
    for i, (mode, vals) in enumerate(data_by_mode.items()):
        style = styles[i % len(styles)]
        cos_vals = np.cos(flatten_vals(vals.get("theta", [])))  # transform
        ax_costh.hist(cos_vals, bins=cos_edges, histtype="step", **style)
    #draw_stats_boxes(ax_costh, data_by_mode, "theta", bin_edges=cos_edges, include_flow=True)
    ax_costh.set_xlabel(r"$\cos\theta$", fontsize=label_fontsize)
    ax_costh.set_ylabel(f"Events / {bin_w_cos:.2f}", fontsize=label_fontsize)
    ax_costh.grid(True, ls='--', alpha=0.7)

    # φ distribution
    for i, (mode, vals) in enumerate(data_by_mode.items()):
        style = styles[i % len(styles)]
        ax_phi.hist(flatten_vals(vals.get("phi", [])), bins=phi_edges, histtype="step", **style)
    draw_stats_boxes(ax_phi, data_by_mode, "phi", bin_edges=phi_edges, include_flow=True)
    ax_phi.set_xlabel(r"$\phi$ [rad]", fontsize=label_fontsize)
    ax_phi.set_ylabel(f"Events / {bin_w_phi:.2f} rad", fontsize=label_fontsize)
    ax_phi.grid(True, ls='--', alpha=0.7)

    plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.35)
    if output_file_prefix:
        plt.savefig(f"{output_file_prefix}_theta_costh_phi_overlay.{file_format}", bbox_inches="tight")
    plt.close(fig)
    # plt.show()
    
    
def plot_jet_differences(delta_eta, delta_theta, delta_phi=None,
                         output_file_prefix=None,
                         file_format='png'):
    """
    Plot Δη, Δθ, and optionally Δφ histograms with fixed bin widths, 
    normalised to events/bin width, with a stats box on each plot.
    """

    label_fontsize = 16
    bin_width = 0.2  # for Δη and Δθ; Δφ can have smaller bins if desired

    # ---- Δη ----
    if delta_eta is not None and len(delta_eta) > 0:
        eta_min = np.floor(min(delta_eta) / bin_width) * bin_width
        eta_max = np.ceil(max(delta_eta) / bin_width) * bin_width
        eta_edges = np.arange(eta_min, eta_max + bin_width, bin_width)

        counts, _ = np.histogram(delta_eta, bins=eta_edges)
        fig_eta, ax_eta = plt.subplots(figsize=(6, 4))
        ax_eta.bar(eta_edges[:-1], counts / bin_width, width=bin_width,
                   color='green', alpha=0.7, edgecolor='gray', linewidth=1.2)
        ax_eta.set_xlabel(r'$\Delta \eta$', fontsize=label_fontsize)
        ax_eta.set_ylabel(f"Events / {bin_width}", fontsize=label_fontsize)
        ax_eta.grid(True, linestyle='--', alpha=0.7)
        annotate_stats(ax_eta, delta_eta, eta_edges)

        plt.tight_layout()
        if output_file_prefix:
            plt.savefig(f"{output_file_prefix}eta_diff.{file_format}", bbox_inches="tight")
            # plt.show()
        plt.close(fig_eta)
    else:
        print("No Δη data")

    # ---- Δθ ----
    if delta_theta is not None and len(delta_theta) > 0:
        th_min = -np.pi
        th_max = np.pi
        th_edges = np.arange(th_min, th_max + bin_width, bin_width)

        counts, _ = np.histogram(delta_theta, bins=th_edges)
        fig_th, ax_th = plt.subplots(figsize=(6, 4))
        ax_th.bar(th_edges[:-1], counts / bin_width, width=bin_width,
                  color='purple', alpha=0.7, edgecolor='gray', linewidth=1.2)
        ax_th.set_xlabel(r'$\Delta \theta$ [rad]', fontsize=label_fontsize)
        ax_th.set_ylabel(f"Events / {bin_width} rad", fontsize=label_fontsize)
        ax_th.grid(True, linestyle='--', alpha=0.7)
        annotate_stats(ax_th, delta_theta, th_edges)

        plt.tight_layout()
        if output_file_prefix:
            plt.savefig(f"{output_file_prefix}theta_diff.{file_format}", bbox_inches="tight")
            # plt.show()
        plt.close(fig_th)
    else:
        print("No Δθ data")

    # ---- Δφ ----
    if delta_phi is not None and len(delta_phi) > 0:
        bin_width_phi = 2*np.pi / 20
        phi_min = -np.pi
        phi_max = np.pi
        phi_edges = np.arange(phi_min, phi_max + bin_width_phi, bin_width_phi)

        counts, _ = np.histogram(delta_phi, bins=phi_edges)
        fig_phi, ax_phi = plt.subplots(figsize=(6, 4))
        ax_phi.bar(phi_edges[:-1], counts / bin_width_phi, width=bin_width_phi,
                   color='orange', alpha=0.7, edgecolor='gray', linewidth=1.2)
        ax_phi.set_xlabel(r'$\Delta \phi$ [rad]', fontsize=16)
        ax_phi.set_ylabel(f"Events / {bin_width_phi:.3f} rad", fontsize=16)
        ax_phi.grid(True, linestyle='--', alpha=0.7)

        # annotate stats in default location
        annotate_stats(ax_phi, delta_phi, phi_edges)

        plt.tight_layout()
        if output_file_prefix:
            plt.savefig(f"{output_file_prefix}phi_diff.{file_format}", bbox_inches="tight")
            # plt.show()
        plt.close(fig_phi)
    else:
        print("No Δφ data")
        
        
#==================================================================
#========================= PARTICLE IDs ===========================
#==================================================================

def parse_summary_output(filename):
    """
    Reads a summary_output file and counts events by unique Initial IDs.
    
    Parameters
    ----------
    filename : str
        Path to the summary_output file
    
    Returns
    -------
    dict
        Dictionary mapping tuple of Initial IDs -> total event count
    """
    pattern = re.compile(r"Initial IDs = ([\d\-]+),([\d\-]+).*Events:\s+(\d+)")
    counts = defaultdict(int)

    with open(filename, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                id1, id2, events = match.groups()
                ids = (int(id1), int(id2))
                counts[ids] += int(events)
    
    return dict(counts)


def plot_initial_ids_counts(counts, what_process, outpath=None):
    """
    Plots event counts by Initial IDs in both linear and log scales side by side.
    
    Parameters
    ----------
    counts : dict
        Dictionary mapping Initial ID pairs to event counts
    outpath : str, optional
        Full path to save the figure. If None, figure is not saved.
    """
    labels = [f"{id1},{id2}" for (id1, id2) in counts.keys()]
    values = list(counts.values())
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    # Linear scale
    axes[0].bar(labels, values)
    axes[0].set_xlabel("Initial IDs")
    axes[0].set_ylabel("Total Events")
    axes[0].set_title("Linear Scale")
    
    if what_process in ["2to2_QCD", "2to2_PhaseSpace"]:
        axes[0].tick_params(axis='x', rotation=45, labelsize = 10)
    
    elif what_process == "2to4":
        axes[0].tick_params(axis='x', rotation=45, labelsize = 8)

    # Log scale
    axes[1].bar(labels, values)
    axes[1].set_xlabel("Initial IDs")
    axes[1].set_ylabel("Total Events")
    axes[1].set_yscale("log")
    axes[1].set_title("Log Scale")
    
    if what_process in ["2to2_QCD", "2to2_PhaseSpace"]:
        axes[1].tick_params(axis='x', rotation=45, labelsize = 10)
    
    elif what_process == "2to4":
        axes[1].tick_params(axis='x', rotation=45, labelsize = 8)

    plt.tight_layout()

    # Save figure if outpath provided
    if outpath is not None:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=300)
    plt.close(fig)
    # plt.show()
    
    
def parse_summary_output_outgoing(filename):
    """
    Reads a summary_output file and counts events by unique Outgoing IDs.
    """
    # Match: "Outgoing IDs = a,b | Events: N"
    pattern = re.compile(r"Outgoing IDs = ([\d\-,]+)\s*\|\s*Events:\s+(\d+)")
    counts = defaultdict(int)

    with open(filename, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                ids_str, events = m.groups()
                ids = tuple(int(x) for x in ids_str.split(","))
                counts[ids] += int(events)

    return dict(counts)


def plot_outgoing_ids_counts(counts, what_process, outpath=None):
    """
    Plot event counts by Outgoing IDs in both linear and log scales side by side.
    
    Parameters
    ----------
    counts : dict
        Dictionary mapping Outgoing ID tuples to event counts
    outpath : str, optional
        Full path to save the figure. If None, figure is not saved.
    """
    labels = [",".join(map(str, k)) for k in counts.keys()]
    values = list(counts.values())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    # Linear scale
    axes[0].bar(labels, values, color="teal", alpha=0.8, edgecolor="black")
    axes[0].set_xlabel("Outgoing IDs")
    axes[0].set_ylabel("Total Events")
    axes[0].set_title("Linear Scale")
    
    if what_process in ["2to2_QCD", "2to2_PhaseSpace"]:
        axes[0].tick_params(axis='x', rotation=45, labelsize = 10)
    
    elif what_process == "2to4":
        axes[0].tick_params(axis='x', rotation=90, labelsize = 6)

    # Log scale
    axes[1].bar(labels, values, color="teal", alpha=0.8, edgecolor="black")
    axes[1].set_xlabel("Outgoing IDs")
    axes[1].set_ylabel("Total Events")
    axes[1].set_yscale("log")
    axes[1].set_title("Log Scale")
    axes[1].tick_params(axis='x', rotation=90, labelsize = 6)
    
    if what_process in ["2to2_QCD", "2to2_PhaseSpace"]:
        axes[1].tick_params(axis='x', rotation=45, labelsize = 10)
    
    elif what_process == "2to4":
        axes[1].tick_params(axis='x', rotation=90, labelsize = 6)

    plt.tight_layout()

    # Save figure if outpath provided
    if outpath is not None:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=300)
    plt.close(fig)
    # plt.show()

    
#======================================================================
#========================= EVENT SHAPE VARS ===========================
#======================================================================

def get_pT(p):
    # p is a 4-vector in the form [E, Px, Py, Pz]
    px, py = p[1], p[2]
    return np.sqrt(px**2 + py**2)


def sort_by_pT_or_E(momenta_list, sort_by='pT'):
    # momenta_list: list of events, each event is a list of 4 4-momenta
    sorted_list = []

    for event in momenta_list:
        if sort_by == 'pT':
            sorted_event = sorted(event, key=lambda p: get_pT(p), reverse=True)
        elif sort_by == 'E':
            sorted_event = sorted(event, key=lambda p: p[0], reverse=True)
        else:
            raise ValueError("sort_by must be either 'pT' or 'E'")
        
        sorted_list.append(sorted_event)
    
    return sorted_list


def normalized_momentum_tensor(particles):
    #Needs to take in three-vectors or else the dot porduct will be 0
    #This definition is from the 2021 paper, sphericity tensor 
    
    T = np.zeros((3, 3))
    
    #Compute the total magnitude (P) and weighted sums for the tensor
    total_magnitude = 0.0
    for p in particles:
        p = np.array(p)  # Ensure p is a numpy array
        magnitude = np.linalg.norm(p)  # |p| = sqrt(px^2 + py^2 + pz^2)
        if magnitude > 0:  # Avoid division by zero for zero vectors
            total_magnitude += magnitude
            for i in range(3):
                for j in range(3):
                    T[i, j] += (p[i] * p[j]) / magnitude  # Weighted by 1/|p|
    
    # Normalize the tensor by total magnitude
    if total_magnitude > 0:  # Avoid division by zero
        T /= total_magnitude
    
    return T


def event_shape_vars(T):
    
    eigenvalues = np.linalg.eigvalsh(T)  # Returns sorted eigenvalues lowest to highest
    
    sphericity = 3/2 * (eigenvalues[0] + eigenvalues[1])
    
    aplanarity = 3/2 * eigenvalues[0]

    Big_Y = (np.sqrt(3) / 2) * ( eigenvalues[1] - eigenvalues[0] )

    C = 3*(eigenvalues[2]*eigenvalues[1] + eigenvalues[2]*eigenvalues[0] + eigenvalues[1]*eigenvalues[0])

    D = 27*(eigenvalues[0]*eigenvalues[1]*eigenvalues[2])
    
    return eigenvalues, sphericity, aplanarity, Big_Y, C, D



# --------------------- THRUST 

def thrust(momentum):

    # Constants
    NSTUDYMIN = 2  # Minimum number of jets required
    CROSSMIN = 1e-10  # Minimum magnitude for cross-product normalization
    
    # Initial values
    NumJets = momentum.shape[0]
    if NumJets < NSTUDYMIN:
        raise ValueError(f"Too few jets: {NumJets}. At least {NSTUDYMIN} jets required.")
    
    pSum = np.sum(momentum, axis=0)  # Total momentum sum
    pMax = np.zeros(4)  # Stores the maximum momentum vector
    eVal1 = 0.0  # Thrust value
    eVec1 = np.zeros(4)  # Thrust axis
    
    # Iterate over pairs of jets to find the optimal reference vector (orthogonal to jet pairs)
    for i in range(NumJets - 1):
        for j in range(i + 1, NumJets):
            # Calculate reference vector orthogonal to two jet momenta
            nRef = np.zeros(4)
            nRef[1:] = np.cross(momentum[i, 1:], momentum[j, 1:])
            norm = max(CROSSMIN, np.linalg.norm(nRef[1:]))
            nRef = nRef / norm
            
            # Test all sign combinations for the momenta contributions
            pPart = np.zeros(4)
            for k in range(NumJets):
                if k not in (i, j):
                    if np.dot(momentum[k, 1:], nRef[1:]) > 0:
                        pPart += momentum[k]
                    else:
                        pPart -= momentum[k]
            
            for sign1, sign2 in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                pFull = pPart + sign1 * momentum[i] + sign2 * momentum[j]
                pFull[0] = np.linalg.norm(pFull[1:])  # Update energy as magnitude of spatial momentum
                if pFull[0] > pMax[0]:
                    pMax = pFull
    
    # Calculate thrust value and axis
    eVal1 = pMax[0] / pSum[0]
    eVec1 = pMax / pMax[0]
    eVec1[0] = 0.0  # The thrust axis has no energy component
    
    # Calculate transverse thrust and related variables
    pTsum = 0
    pTdot = 0
    pTcross = 0
    
    for i in range(NumJets):
        pTv = momentum[i, 1:3]  # Transverse components (px, py)
        n = eVec1[1:3]  # Transverse part of the thrust axis
        dot_product = np.abs(np.dot(pTv, n))
        cross_product = np.linalg.norm(np.cross(pTv, n))
        pT = np.linalg.norm(pTv)
        
        pTdot += dot_product
        pTcross += cross_product
        pTsum += pT
    
    Tperp = pTdot / pTsum
    Tm = pTcross / pTsum
    tauperp = 1 - Tperp
    
    return eVal1, eVec1, Tperp, Tm, tauperp


#-------------------- BIPLANARITY 

SORT_METHODS = ['pT', 'E']
BIPLANARITY_MODES = [0, 1, 2]

# =============================
# Biplanarity Core Functions
# =============================

# From testing mode 1 with pT ordering in the lab is the best one to use.
def biplanarity(pUnordered, mode=0, sort_by='E'):
    eps = 1e-5
    N = len(pUnordered)
    if N < 4:
        return -1 
        # raise ValueError("Need at least 4 momenta to compute biplanarity")

    if sort_by == 'pT':
        vals = [get_pT(p_i) for p_i in pUnordered]
    elif sort_by == 'E':
        vals = [p_i[0] for p_i in pUnordered]
    else:
        raise ValueError("sort_by must be 'pT' or 'E'")

    index = (-np.array(vals)).argsort()
    p = [pUnordered[i] for i in index[:4]]

    if mode == 0:
        n1 = np.cross(p[0][1:], p[2][1:])
        n2 = np.cross(p[1][1:], p[3][1:])
    elif mode == 1:
        n1 = np.cross(p[0][1:], p[3][1:])
        n2 = np.cross(p[1][1:], p[2][1:])
    elif mode == 2:
        n1 = np.cross(p[0][1:], p[1][1:])
        n2 = np.cross(p[2][1:], p[3][1:])
    else:
        raise ValueError("mode must be 0, 1, or 2")

    norm1 = np.linalg.norm(n1)
    norm2 = np.linalg.norm(n2)
    if norm1 < eps or norm2 < eps:
        return 0.0

    B = abs(np.dot(n1, n2)) / (norm1 * norm2)

    if B < 0 or B > 1 + eps:
        print("Biplanarity out of range:", B)
        B = min(max(B, 0), 1)

    return B

def apply_biplanarity_tolerance(B, tol=1e-14):
    B = np.array(B)
    B[np.abs(B - 1) < tol] = 1
    return B


# =============================
# Biplanarity Evaluation Utils
# =============================

def compute_all_biplanarity(events, label=""):
    print(f"\n====== {label} Events ======")
    for sort_by in SORT_METHODS:
        print(f"\n-- Sorted by: {sort_by} --")
        for mode in BIPLANARITY_MODES:
            bvals = []
            for event in events:
                try:
                    bval = biplanarity(event, sort_by=sort_by, mode=mode)
                    bvals.append(bval)
                except Exception as e:
                    bvals.append(f"Error: {e}")
            print(f"mode {mode}: {bvals}")

# This diff_momentum was changes so make sure it is 
def compute_biplanarity_from_lhe_files(filenames, sort_by, mode):
    B_vals = []
    for filename in filenames:
        cross_section, event_data = read_lhe_file(filename)
        if event_data is None:
            print(f"Warning: No event data found in {filename}")
            continue

        four_momenta = [particles for particles, _, _ in event_data]
        three_momenta = diff_momentum(four_momenta, threeD=True)

        _, _, _, _, _, _, _, _, _, B = calc_EventVars(
            four_momenta, three_momenta, sort_by=sort_by, mode=mode
        )
        B_vals.extend(np.concatenate([np.array(vals).flatten() for vals in B]))

    return B_vals

def sanitize_filename(name):
    return re.sub(r'[^\w\-_.]', '_', name)


# =============================
# Biplanarity Plotting + LaTeX
# =============================

def plot_biplanarity_histograms(B_data_dict, title_prefix, save_plots=False, energy_label=None, frame=None):
    bins = np.linspace(0, 1, 70)
    colors = ['black', 'orange', 'purple']  

    for sort_by in SORT_METHODS:
        plt.figure(figsize=(18, 5))
        #plt.suptitle(f"{frame} Frame — Sorted by {sort_by}", fontsize=16)

        max_height = 0
        hist_data = {}

        for mode in BIPLANARITY_MODES:
            B_vals = B_data_dict[(mode, sort_by)]
            total_events = len(B_vals)
            weights = np.ones_like(B_vals) / total_events
            hist_vals, _ = np.histogram(B_vals, bins=bins, weights=weights)
            hist_data[mode] = (B_vals, hist_vals)
            max_height = max(max_height, max(hist_vals))

        y_max = 1.1 * max_height

        for i, mode in enumerate(BIPLANARITY_MODES, start=1):
            B_vals, _ = hist_data[mode]
            total_events = len(B_vals)
            last_bin_lower_edge = bins[-2]
            count_in_last_bin = np.sum(np.array(B_vals) >= last_bin_lower_edge)
            percent_in_last_bin = 100.0 * count_in_last_bin / total_events

            plt.subplot(1, 3, i)
            plt.hist(B_vals, bins=bins, weights=np.ones_like(B_vals) / total_events,
            histtype='step', edgecolor=colors[i-1], linewidth=1.5)

            plt.xlabel("Biplanarity")
            plt.ylabel("Normalized Frequency")
            plt.ylim(0, y_max)

            info_text = (
                f"mode={mode}\n"
                f"Total: {total_events}\n"
                f">= {last_bin_lower_edge:.2f}: "
                f"{count_in_last_bin} ({percent_in_last_bin:.2f}%)"
            )
            plt.title(info_text, fontsize=10)
            plt.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()

        # Build filename
        modes_str = "allmodes" if len(BIPLANARITY_MODES) == 3 else f"mode{BIPLANARITY_MODES[0]}"
        plot_filename = f"{modes_str}_{sort_by}_{frame}_bip_single.png"

        if save_plots:
            output_dir = os.path.join("Plots", "BipStudy")
            os.makedirs(output_dir, exist_ok=True)

            save_path = os.path.join(output_dir, plot_filename)

            plt.savefig(save_path, dpi=300)
            print(f"Plot saved as {save_path}")
        # plt.close(fig)
        # plt.show()


def plot_biplanarity_overlay(B_data_dict, title_prefix, save_plots=False, energy_label=None, frame=None):
    bins = np.linspace(0, 1, 70)
    color_map = ['black', 'orange', 'purple']

    for sort_by in SORT_METHODS:
        fig, ax = plt.subplots(figsize=(9, 6))
        #plt.title(f"{frame} Frame — Sorted by {sort_by}", fontsize=14)

        legend_entries = []

        # Use total events from mode 0 only (assuming same for all modes)
        total_events = len(B_data_dict[(0, sort_by)])
        legend_entries.append(f"Total Events: {total_events}")
        legend_entries.append("")  # blank line

        for mode in BIPLANARITY_MODES:
            B_vals = B_data_dict[(mode, sort_by)]
            total_events_mode = len(B_vals)
            weights = np.ones_like(B_vals) / total_events_mode

            hist_vals, bin_edges = np.histogram(B_vals, bins=bins, weights=weights)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            ax.step(bin_centers, hist_vals, where='mid', color=color_map[mode], linewidth=1.5)

            last_bin_lower_edge = bins[-2]
            count_in_last_bin = np.sum(np.array(B_vals) >= last_bin_lower_edge)
            percent_in_last_bin = 100.0 * count_in_last_bin / total_events_mode

            legend_entries.append(
                f"Mode {mode}\nLast Bin: {count_in_last_bin} ({percent_in_last_bin:.2f}%)"
            )

        ax.set_xlabel("Biplanarity")
        ax.set_ylabel("Normalized Frequency")
        ax.grid(True, linestyle="--", alpha=0.5)

        from matplotlib.patches import Patch
        handles = [Patch(facecolor='none', edgecolor='none', label=legend_entries[0])]  # Total Events at top
        handles.append(Patch(facecolor='none', edgecolor='none', label=""))  # blank line

        for mode, color in zip(BIPLANARITY_MODES, color_map):
            mode_text = legend_entries[mode + 2]
            handles.append(Patch(facecolor='none', edgecolor=color, linewidth=1.5, label=mode_text))

        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, frameon=True)

        plt.tight_layout(rect=[0, 0, 0.8, 1])

        modes_str = "allmodes" if len(BIPLANARITY_MODES) == 3 else f"mode{BIPLANARITY_MODES[0]}"
        plot_filename = f"{modes_str}_{sort_by}_{frame}_bip_overlay.png"

        if save_plots:
            output_dir = os.path.join("Plots", "BipStudy")
            os.makedirs(output_dir, exist_ok=True)

            save_path = os.path.join(output_dir, plot_filename)

            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"Overlay plot saved as {save_path}")
        # plt.close(fig)
        # plt.show()


def generate_biplanarity_latex_table(B_data_dict, table_caption, table_label):
    bins = np.linspace(0, 1, 70)
    last_bin_lower_edge = bins[-2]

    for sort_by in SORT_METHODS:
        print(f"\n\\begin{{table}}[h!]")
        print(f"\\centering")
        print(f"\\begin{{tabular}}{{|c|c|}}")
        print(f"\\hline")
        print(f"\\textbf{{Mode}} & \\textbf{{Final Bin (\\%)}} \\\\")
        print(f"\\hline")

        for mode in BIPLANARITY_MODES:
            B_vals = B_data_dict[(mode, sort_by)]
            total_events = len(B_vals)

            count_in_last_bin = np.sum(np.array(B_vals) >= last_bin_lower_edge)
            percent_in_last_bin = 100.0 * count_in_last_bin / total_events

            print(f"{mode} & {percent_in_last_bin:.2f}\\% \\\\")
            print(f"\\hline")

        print(f"\\end{{tabular}}")
        print(f"\\caption{{{table_caption}}}")
        print(f"\\label{{{table_label}}}")
        print(f"\\end{{table}}\n")


#======================================================================
#========================= EVENT SHAPE CALC ===========================
#======================================================================

def calc_EventVars(four_mom, spatial_mom, sort_by='pT', mode=None, verbose=False):
    S, A, S_T, Y, C, D = [], [], [], [], [], []
    Thrust_T, Thrust_m, tau, B = [], [], [], []  # tau = 1 - Thrust_T

    if mode not in [0, 1, 2]:
        raise ValueError("`mode` must be one of: 0, 1, 2.")

    for i, (four_vec, spatial_vec) in enumerate(zip(four_mom, spatial_mom)):
        # Convert to arrays
        four_vec = np.array(four_vec)
        spatial_vec = np.array(spatial_vec)

        # General event shape tensor
        tensor = normalized_momentum_tensor(spatial_vec)
        eigenvalues, sphericity, aplanarity, Y_val, C_val, D_val = event_shape_vars(tensor)

        eigenvalues = apply_tolerance(eigenvalues, tol=1e-15)
        sphericity = apply_tolerance(sphericity, tol=1e-15)
        aplanarity = apply_tolerance(aplanarity, tol=1e-15)
        Y_val = apply_tolerance(Y_val, tol=1e-15)
        C_val = apply_tolerance(C_val, tol=1e-15)
        D_val = apply_tolerance(D_val, tol=1e-15)

        S.append(sphericity)
        A.append(aplanarity)
        Y.append(Y_val)
        C.append(C_val)
        D.append(D_val)

        # Transverse tensor variables
        transverse_tensor, trans_evals, trans_sphr = normalized_momentum_tensor(spatial_vec)
        trans_sphr = apply_tolerance(trans_sphr, tol=1e-15)
        S_T.append(trans_sphr)

        # Thrust variables
        eVal1_, eVec1_, T_perp, T_m, tau_perp = thrust(four_vec)

        T_perp = apply_tolerance(T_perp, tol=1e-15)
        T_m = apply_tolerance(T_m, tol=1e-15)
        tau_perp = apply_tolerance(tau_perp, tol=1e-15)

        Thrust_T.append(T_perp)
        Thrust_m.append(T_m)
        tau.append(tau_perp)

        # Updated biplanarity call using mode
        B_val = biplanarity(four_vec, sort_by=sort_by, mode=mode)
        B.append(B_val)

        if verbose:
            print(f"\033[1mEvent {i + 1}:\033[0m")
            print("Normalized momentum tensor:\n", tensor)
            print("Ordered eigenvalues:", eigenvalues)
            print("Event variables:")
            print(f"  Sphericity: {sphericity}")
            print(f"  Aplanarity: {aplanarity}")
            print(f"  Y: {Y_val}, C: {C_val}, D: {D_val}")
            print("Transverse sphericity:", trans_sphr)
            print(f"Thrust (T_perp): {T_perp}, Minor: {T_m}, Tau: {tau_perp}")
            print(f"Biplanarity ({sort_by}, mode={mode}): {B_val}")
            print()

    return S, A, S_T, Y, C, D, Thrust_T, Thrust_m, tau, B


#=======================================================================
#========================= EVENT SHAPE PLOTTING ========================
#=======================================================================

def save_individual_plot(data, xlabel, ylabel, title, filename, file_format='pdf', label_fontsize=16, tick_fontsize=16, xlim=None, ylim=None):
    """Helper function to save individual plots."""
    fig = plt.figure(figsize=(7, 6))
    plt.hist(data, bins=50, color='blue', alpha=0.7, edgecolor='gray', linewidth=1.2)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.title(title)
    plt.tick_params(axis='both', labelsize=tick_fontsize)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    plt.yscale('log')
    
    # Set axis limits if provided
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
        
    plt.savefig(f"{filename}.{file_format}", bbox_inches='tight')
    plt.close(fig)


def plot_event_variables(S, A, S_T, Y, C, D, Thrust_T, Thrust_m, tau, B, output_file_prefix=None, file_format='pdf', show_group=True):
    label_fontsize = 16
    tick_fontsize = 16
    marker_size = 20
    title_fontsize = 20  # Universal title font size

    # Define axis limits for specific variables
    limits = {
        'S': (0, 1),        # Sphericity range
        'A': (0, 0.5),      # Aplanarity range
        'B': (0, 1.05),     # Biplanarity range, needs to be a little larger so I can get the bin displayed
    }

    # Grouped plot
    if show_group:
        fig1, axes = plt.subplots(1, 3, figsize=(21, 6))
        fig1.suptitle('Event Shape Variables', fontsize=30)  # Universal title size
        
        # Sphericity Histogram
        axes[0].hist(S, bins=50, color='blue', alpha=0.7, edgecolor='gray', linewidth=1.2)
        axes[0].set_xlabel('S', fontsize=label_fontsize)
        axes[0].set_ylabel('Frequency', fontsize=label_fontsize)
        axes[0].set_title(r"Sphericity")
        axes[0].tick_params(axis='both', labelsize=tick_fontsize)
        axes[0].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        axes[0].set_yscale('log')
        axes[0].set_xlim(limits['S'])  # Apply the axis limit for S

        # Aplanarity Histogram
        axes[1].hist(A, bins=50, color='blue', alpha=0.7, edgecolor='gray', linewidth=1.2)
        axes[1].set_xlabel('A', fontsize=label_fontsize)
        axes[1].set_ylabel('Frequency', fontsize=label_fontsize)
        axes[1].set_title(r"Aplanarity")
        axes[1].tick_params(axis='both', labelsize=tick_fontsize)
        axes[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        axes[1].set_yscale('log')
        axes[1].set_xlim(limits['A'])  # Apply the axis limit for A

        # Scatter Plot
        axes[2].scatter(S, Y, color='purple', alpha=0.6, s=marker_size, zorder=2)
        axes[2].set_xlabel('S', fontsize=label_fontsize)
        axes[2].set_ylabel('Y', fontsize=label_fontsize)
        axes[2].set_title(r"Y vs. S")
        axes[2].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        axes[2].grid(zorder=1)

        # Set the title font size universally for all subplots
        for ax in axes:
            ax.title.set_fontsize(title_fontsize)

        plt.tight_layout()

        if output_file_prefix:
            plt.savefig(f"{output_file_prefix}_SAY.{file_format}", bbox_inches='tight')
        plt.close(fig)
        # plt.show()

        # Create the next figures
        # Figure 2: C, D, Transverse Sphericity
        fig2, axes = plt.subplots(1, 3, figsize=(21, 6))
        axes[0].hist(C, bins=50, color='blue', alpha=0.7, edgecolor='gray', linewidth=1.2)
        axes[0].set_xlabel('C', fontsize=label_fontsize)
        axes[0].set_ylabel('Frequency', fontsize=label_fontsize)
        axes[0].set_title(r"C Histogram")
        axes[0].tick_params(axis='both', labelsize=tick_fontsize)
        axes[0].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        axes[0].set_yscale('log')

        axes[1].hist(D, bins=50, color='blue', alpha=0.7, edgecolor='gray', linewidth=1.2)
        axes[1].set_xlabel('D', fontsize=label_fontsize)
        axes[1].set_ylabel('Frequency', fontsize=label_fontsize)
        axes[1].set_title(r"D Histogram")
        axes[1].tick_params(axis='both', labelsize=tick_fontsize)
        axes[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        axes[1].set_yscale('log')

        axes[2].hist(S_T, bins=50, color='blue', alpha=0.7, edgecolor='gray', linewidth=1.2)
        axes[2].set_xlabel(r"$S_{\perp}$", fontsize=label_fontsize)
        axes[2].set_ylabel('Frequency', fontsize=label_fontsize)
        axes[2].set_title(r"Transverse Sphericity")
        axes[2].tick_params(axis='both', labelsize=tick_fontsize)
        axes[2].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        axes[2].set_yscale('log')

        # Set title font size for all subplots
        for ax in axes:
            ax.title.set_fontsize(title_fontsize)

        plt.tight_layout()
        if output_file_prefix:
            plt.savefig(f"{output_file_prefix}_CDST.{file_format}", bbox_inches='tight')
        plt.close(fig)
        # plt.show()

        # Figure 3: Thrust approximations
        fig3, axes = plt.subplots(1, 3, figsize=(21, 6))
        axes[0].hist(Thrust_T, bins=50, color='blue', alpha=0.7, edgecolor='gray', linewidth=1.2)
        axes[0].set_xlabel(r"$T_{\perp}$", fontsize=label_fontsize)
        axes[0].set_ylabel('Frequency', fontsize=label_fontsize)
        axes[0].set_title(r"Transverse Thrust")
        axes[0].tick_params(axis='both', labelsize=tick_fontsize)
        axes[0].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        axes[0].set_yscale('log')

        axes[1].hist(Thrust_m, bins=50, color='blue', alpha=0.7, edgecolor='gray', linewidth=1.2)
        axes[1].set_xlabel(r"$T_m$", fontsize=label_fontsize)
        axes[1].set_ylabel('Frequency', fontsize=label_fontsize)
        axes[1].set_title(r"Minor Transverse Thrust")
        axes[1].tick_params(axis='both', labelsize=tick_fontsize)
        axes[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        axes[1].set_yscale('log')

        axes[2].hist(tau, bins=50, color='blue', alpha=0.7, edgecolor='gray', linewidth=1.2)
        axes[2].set_xlabel(r"$\tau_{\perp}$", fontsize=label_fontsize)
        axes[2].set_ylabel('Frequency', fontsize=label_fontsize)
        axes[2].set_title(r"Tau Histogram")
        axes[2].tick_params(axis='both', labelsize=tick_fontsize)
        axes[2].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        axes[2].set_yscale('log')

        # Set title font size for all subplots
        for ax in axes:
            ax.title.set_fontsize(title_fontsize)

        plt.tight_layout()
        if output_file_prefix:
            plt.savefig(f"{output_file_prefix}_Thrusts.{file_format}", bbox_inches='tight')
        plt.close(fig)
        # plt.show()

        # Figure 4: Biplanarity
        fig4, axes = plt.subplots(1, 3, figsize=(21, 6))

        axes[0].hist(B, bins=50, color='blue', alpha=0.7, edgecolor='gray', linewidth=1.2)
        axes[0].set_xlabel(r"B", fontsize=label_fontsize)
        axes[0].set_ylabel('Frequency', fontsize=label_fontsize)
        axes[0].set_title(r"B Histogram", fontsize=title_fontsize)
        axes[0].tick_params(axis='both', labelsize=tick_fontsize)
        axes[0].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        axes[0].set_yscale('log')
        axes[0].set_xlim(limits['B'])  # Apply the axis limit for B

        axes[1].axis("off")
        axes[2].axis("off")

        plt.tight_layout()
        if output_file_prefix:
            plt.savefig(f"{output_file_prefix}_B.{file_format}", bbox_inches='tight')
        plt.close(fig)
        # plt.show()

    else:
        # If save_group is False, save individual plots using the helper function
        save_individual_plot(S, 'S', 'Frequency', 'Sphericity', f"{output_file_prefix}_S", file_format, label_fontsize, tick_fontsize, xlim=limits['S'])
        save_individual_plot(A, 'A', 'Frequency', 'Aplanarity', f"{output_file_prefix}_A", file_format, label_fontsize, tick_fontsize, xlim=limits['A'])
        save_individual_plot(S, 'S', 'Y', 'Y vs. S', f"{output_file_prefix}_Scatter", file_format, label_fontsize, tick_fontsize)
        save_individual_plot(C, 'C', 'Frequency', 'C Histogram', f"{output_file_prefix}_C", file_format, label_fontsize, tick_fontsize)
        save_individual_plot(D, 'D', 'Frequency', 'D Histogram', f"{output_file_prefix}_D", file_format, label_fontsize, tick_fontsize)
        save_individual_plot(S_T, r"$S_{\perp}$", 'Frequency', 'Transverse Sphericity', f"{output_file_prefix}_ST", file_format, label_fontsize, tick_fontsize)
        save_individual_plot(Thrust_T, r"$T_{\perp}$", 'Frequency', 'Transverse Thrust', f"{output_file_prefix}_ThrustT", file_format, label_fontsize, tick_fontsize)
        save_individual_plot(Thrust_m, r"$T_m$", 'Frequency', 'Minor Transverse Thrust', f"{output_file_prefix}_Thrust_m", file_format, label_fontsize, tick_fontsize)
        save_individual_plot(tau, r"$\tau_{\perp}$", 'Frequency', 'Tau', f"{output_file_prefix}_Tau", file_format, label_fontsize, tick_fontsize)
        save_individual_plot(B, 'B', 'Frequency', 'B Histogram', f"{output_file_prefix}_B", file_format, label_fontsize, tick_fontsize, xlim=limits['B'])

        plt.show()


# Note that the individual Y vs. S plot is not a scatter plot. It is only a scatter when I do the grouped = True


#===========================================================
#========================= READ LHE ========================
#===========================================================

def read_lhe_file(file_path):
    with open(file_path, "r") as file:
        inside_event = False
        inside_init = False
        events = []  # Store all events
        cross_section = None  # Store cross-section
        
        for line in file:
            line = line.strip()

            # Extract cross-section from <init> section
            if line.startswith("<init>"):
                inside_init = True
                continue

            if line.startswith("</init>"):
                inside_init = False
                continue

            if inside_init:
                parts = line.split()
                if len(parts) >= 2:
                    cross_section = float(parts[0])  # Extract first float in the second line
                continue

            # Start of a new event
            if line.startswith("<event>"):
                inside_event = True
                final_state_particles = []  # Store final-state particles only
                xa, xb = None, None  # Initialize xa and xb for this event
                continue

            if line.startswith("</event>"):
                inside_event = False
                # Store only the last 4 final-state particles along with xa, xb
                if len(final_state_particles) >= 4 and xa is not None and xb is not None:
                    events.append((final_state_particles[-4:], xa, xb))
                continue

            # Process event data
            if inside_event:
                particle_data = line.split()
                
                # Extract final-state particles (status = 1)
                if len(particle_data) >= 10:
                    status = int(particle_data[1])  # Particle status
                    if status == 1:  # Final-state particle
                        final_state_particles.append([
                            float(particle_data[9]),  # E
                            float(particle_data[6]),  # Px
                            float(particle_data[7]),  # Py
                            float(particle_data[8]),  # Pz
                        ])
                
                # Extract xa and xb from the PDF line
                if line.startswith("#pdf"):
                    pdf_data = line.split()
                    xa, xb = float(pdf_data[3]), float(pdf_data[4])

    return cross_section, events  # Now events contain (final_state_particles, xa, xb)



def read_lhe_grouped_by_lprup(file_path):
    """
    Parse an LHE file and group events by IDPRUP (lprup).
    Returns a dict: { lprup: {"cross_section": float|None, "error": float|None,
                             "maxw": float|None, "events": [ {...}, ... ] } }
    Each event is stored as a dict with keys: "NUP", "final_state" (list of (E,px,py,pz)),
    "xa", "xb", and optionally other header fields.
    """

    lprup_data = {}

    with open(file_path, "r") as f:
        lines = iter(f)
        inside_init = False

        for raw in lines:
            line = raw.strip()

            # --- INIT BLOCK ---
            if line == "<init>":
                inside_init = True
                # skip the init header line (beam ids, energies, PDF ids, IDWTUP, NPRUP)
                try:
                    next(lines)
                except StopIteration:
                    break
                # now read the subsequent init lines until </init>
                for raw_init in lines:
                    linit = raw_init.strip()
                    if linit == "</init>":
                        inside_init = False
                        break
                    parts = linit.split()
                    # Expect lines like: xsec err maxw lprup
                    if len(parts) >= 4:
                        try:
                            xsec = float(parts[0])
                            err = float(parts[1])
                            maxw = float(parts[2])
                            lprup = int(parts[3])
                        except ValueError:
                            continue
                        lprup_data.setdefault(lprup, {
                            "cross_section": xsec,
                            "error": err,
                            "maxw": maxw,
                            "events": []
                        })
                continue

            # --- EVENT BLOCK ---
            if line == "<event>":
                # Read the header line right after <event>
                try:
                    header_raw = next(lines)
                except StopIteration:
                    break
                header = header_raw.strip().split()
                # Header expected format: NUP IDPRUP XWGTUP SCALUP AQED AQCD
                # So IDPRUP is header[1] (if present)
                try:
                    NUP = int(header[0])
                except (IndexError, ValueError):
                    NUP = None
                try:
                    IDPRUP = int(header[1])
                except (IndexError, ValueError):
                    IDPRUP = None

                final_state = []
                xa = xb = None

                # Read event body until </event>
                for raw_ev_line in lines:
                    ev_line = raw_ev_line.strip()
                    if ev_line == "</event>":
                        break
                    # pdf line contains xa and xb at positions 3 and 4: "#pdf id1 id2 xa xb ..."
                    if ev_line.startswith("#pdf"):
                        pdf_parts = ev_line.split()
                        if len(pdf_parts) >= 5:
                            try:
                                xa = float(pdf_parts[3])
                                xb = float(pdf_parts[4])
                            except ValueError:
                                xa = xb = None
                        continue
                    # particle lines: expect at least 10 columns; status is column 1
                    parts = ev_line.split()
                    if len(parts) >= 10:
                        try:
                            status = int(parts[1])
                        except ValueError:
                            continue
                        if status == 1:
                            # E : parts[9], px: parts[6], py: parts[7], pz: parts[8]
                            try:
                                E  = float(parts[9])
                                px = float(parts[6])
                                py = float(parts[7])
                                pz = float(parts[8])
                                final_state.append((E, px, py, pz))
                            except ValueError:
                                pass

                # Ensure there is an entry for this IDPRUP (from init or create placeholder)
                if IDPRUP not in lprup_data:
                    lprup_data[IDPRUP] = {
                        "cross_section": None,
                        "error": None,
                        "maxw": None,
                        "events": []
                    }

                # Store the event (you can change to store only last 4 final-state particles)
                lprup_data[IDPRUP]["events"].append({
                    "NUP": NUP,
                    "final_state": final_state,
                    "xa": xa,
                    "xb": xb,
                    # optionally store the full header if you want:
                    "event_header": header
                })
                continue

    return lprup_data


#======================================================================
#========================= INVARIANT MASS =============================
#======================================================================

def invar_mass(event):
    total_momentum = np.sum(event, axis=0)  # Sum over all four-momenta
    E_tot, px_tot, py_tot, pz_tot = total_momentum
    M2 = E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)
    return np.sqrt(M2)

# Background model function
def bg3(m, p0, p1, p2):
    sqrts = 13.6  # Center-of-mass energy in TeV
    x = m / sqrts
    return p0 * np.power(1 - x, p1) * np.power(x, p2)


def plottt(sphericity, aplanarity, sphericity_transverse, Y_values,
                         C_values, D_values, Thrust_T_values, Thrust_m_values,
                         tau_values, B_values, save=False, directory=None):
    """
    Plots the event variables and optionally saves the images.

    Parameters:
    - sphericity, aplanarity, etc.: The event variables to plot
    - save (bool): If True, saves the plot to a file, otherwise displays it
    - directory (str): Directory name for naming the saved files
    """
    # List of event variables to plot
    event_vars = [
        ("Sphericity", sphericity),
        ("Aplanarity", aplanarity),
        ("Sphericity Transverse", sphericity_transverse),
        ("Y Values", Y_values),
        ("C Values", C_values),
        ("D Values", D_values),
        ("Thrust T Values", Thrust_T_values),
        ("Thrust m Values", Thrust_m_values),
        ("Tau Values", tau_values),
        ("B Values", B_values)
    ]

    # Plot each variable
    for title, data in event_vars:
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=50, edgecolor='black', alpha=0.7)
        plt.title(f'{title} for {directory}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        # Save or show the plot based on the flag
        if save:
            plt.savefig(f'{directory}_{title.replace(" ", "_")}.png')
            print(f"Saved plot: {directory}_{title.replace(' ', '_')}.png")
        else:
            plt.show()

        plt.close()  # Close the plot to avoid overlapping in future iterations