import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# =====================================================================
# MODULE 1: PHYSICAL PARAMETERS INITIALIZATION
# =====================================================================
print("="*60)
print("🚀 INITIALIZING MODULE 1: COUPLING CONSTANTS & CHARACTERISTIC SCALES")
print("="*60)

# System Setup Parameters
m_b_amu = 132.90545
N_b = 10000
m_f_amu = 6.015122
N_f = 50000

nu_rb = 80.0
nu_zb = 6.0
nu_rf = 320.0
nu_zf = 24.0

a_bb = 280.0
a_bf_initial = 0.0

# Physical Constants & Scale Conversion Calculations
hbar = 1.0545718e-34
u_atomic_unit = 1.66053906e-27
kB = 1.380649e-23

m_b_kg = m_b_amu * u_atomic_unit
m_f_kg = m_f_amu * u_atomic_unit

omega_rb = 2 * np.pi * nu_rb
osc_r = np.sqrt(hbar / (m_b_kg * omega_rb)) # Radial Harmonic Length Scale
osc_t = 1.0 / omega_rb                      # Characteristic Time Scale
osc_E = hbar * omega_rb                     # Characteristic Energy Scale
density_conversion = 1.0 / ((osc_r * 1e6)**3)

mass_f_scaled = m_f_kg / m_b_kg
omega_zb_scaled = (2 * np.pi * nu_zb) / omega_rb
omega_rf_scaled = (2 * np.pi * nu_rf) / omega_rb
omega_zf_scaled = (2 * np.pi * nu_zf) / omega_rb

g_bb = 4 * np.pi * (a_bb * 5.2917721e-11) / osc_r
m_R = (m_b_kg * m_f_kg) / (m_b_kg + m_f_kg)
g_bf_initial = 2 * np.pi * (a_bf_initial * 5.2917721e-11) * (m_b_kg / m_R) / osc_r
A_fermi = 0.5 * (1.0 / mass_f_scaled) * (3 * np.pi**2)**(2/3)

print("✅ Module 1 Variables Locked.\n")


# =====================================================================
# MODULE 2: GRID DEFINITION & GROUND STATE RELAXATION
# =====================================================================
print("="*60)
print("🧘 INITIALIZING MODULE 2: RELAXATION SPACE MESH & EQUILIBRIUM SOLVER")
print("="*60)

# STABILIZATION UPDATE: Tightening basic grids globally to cleanly resolve collapse spikes
Lr_fixed = 25.0    # Fixed Box Radius (μm)
Lz_fixed = 250.0   # Fixed Box Half-Length (μm)
delta_r = 0.3      # Optimized Radial Step Size (μm)
delta_z = 0.8      # Optimized Axial Step Size (μm)
dt_ground_ms = 0.002
max_iterations = 2000

# Coordinate Transformation & Meshgrid Mapping
dr = (delta_r * 1e-6) / osc_r
dz = (delta_z * 1e-6) / osc_r
Lr_scaled = (Lr_fixed * 1e-6) / osc_r
Lz_scaled = (Lz_fixed * 1e-6) / osc_r
dt_g = (dt_ground_ms * 1e-3) / osc_t

r_vec = np.arange(dr/2, Lr_scaled, dr)
z_vec = np.arange(-Lz_scaled, Lz_scaled, dz)
Nr, Nz = len(r_vec), len(z_vec)
R, Z = np.meshgrid(r_vec, z_vec, indexing='ij')

V_b = 0.5 * (R**2 + (omega_zb_scaled * Z)**2)
V_f = 0.5 * mass_f_scaled * ((omega_rf_scaled * R)**2 + (omega_zf_scaled * Z)**2)

# Boundary Sponge Filter Absorber Layer
sponge = np.ones((Nr, Nz))
br, bz = 0.94 * Lr_scaled, 0.94 * Lz_scaled
sponge[R > br] *= np.exp(-40.0 * ((R[R > br] - br) / (Lr_scaled - br))**4)
sponge[np.abs(Z) > bz] *= np.exp(-40.0 * ((np.abs(Z[np.abs(Z) > bz]) - bz) / (Lz_scaled - bz))**4)

def norm_g(psi):
    return 2 * np.pi * np.sum(np.abs(psi)**2 * R * dr * dz)

# Initialization Guess Wavefunctions
psi_b = np.exp(-(R**2 / 120.0 + Z**2 / 1200.0))
psi_f = np.exp(-(R**2 / 140.0 + Z**2 / 1400.0))
psi_b *= np.sqrt(N_b / norm_g(psi_b))
psi_f *= np.sqrt(N_f / norm_g(psi_f))

mu_b_old, mu_f_old = 0, 0
mid_z_idx = int(Nz / 2)

print("Processing imaginary-time mixture relaxation loops...")
for iteration in range(max_iterations):
    rho_b = np.abs(psi_b)**2
    rho_f = np.abs(psi_f)**2
    ramp = 1.0 if iteration > 200 else (iteration / 200.0)
    mu_fermi_term = A_fermi * np.maximum(rho_f, 1e-15)**(2/3)

    U_b = V_b + (g_bb * rho_b + g_bf_initial * rho_f) * ramp
    U_f = V_f + mu_fermi_term + (g_bf_initial * rho_b) * ramp

    psi_b *= np.exp(-np.clip(U_b, -45/dt_g, 45/dt_g) * dt_g)
    psi_f *= np.exp(-np.clip(U_f, -45/dt_g, 45/dt_g) * dt_g)

    # 2D Finite Difference Spatial Laplacian Steps
    for psi, mass_c in [(psi_b, 1.0), (psi_f, mass_f_scaled)]:
        lap = np.zeros_like(psi)
        lap[1:-1, 1:-1] = (
            (psi[2:, 1:-1] - 2*psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dr**2 +
            (1.0 / R[1:-1, 1:-1]) * (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2*dr) +
            (psi[1:-1, 2:] - 2*psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dz**2
        )
        psi += (0.5 / mass_c) * lap * dt_g

    psi_b *= sponge; psi_f *= sponge
    psi_b *= np.sqrt(N_b / norm_g(psi_b))
    psi_f *= np.sqrt(N_f / norm_g(psi_f))

    if iteration % 200 == 0 and iteration > 200:
        if abs(U_b[0, mid_z_idx] - mu_b_old) < 1e-7 and abs(U_f[0, mid_z_idx] - mu_f_old) < 1e-7:
            print(f" --> Convergence criteria reached early at step iteration {iteration}.")
            break
        mu_b_old, mu_f_old = U_b[0, mid_z_idx], U_f[0, mid_z_idx]

# Ground State Integral Energy Analysis
E_pot_int = np.sum((V_b*np.abs(psi_b)**2 + V_f*np.abs(psi_f)**2 + 0.5*g_bb*np.abs(psi_b)**4 + g_bf_initial*np.abs(psi_b)**2*np.abs(psi_f)**2 + 0.6*A_fermi*np.maximum(np.abs(psi_f)**2, 1e-15)**(5/3)) * R)
E_pot_SI = E_pot_int * 2 * np.pi * dr * dz * osc_E

lap_b = np.zeros_like(psi_b)
lap_b[1:-1, 1:-1] = ((psi_b[2:, 1:-1] - 2*psi_b[1:-1, 1:-1] + psi_b[:-2, 1:-1])/dr**2 + (1.0/R[1:-1, 1:-1])*(psi_b[2:, 1:-1] - psi_b[:-2, 1:-1])/(2*dr) + (psi_b[1:-1, 2:] - 2*psi_b[1:-1, 1:-1] + psi_b[1:-1, :-2])/dz**2)
E_kin_b_SI = -0.5 * 2 * np.pi * np.sum(np.conj(psi_b) * lap_b * R * dr * dz) * osc_E

lap_f = np.zeros_like(psi_f)
lap_f[1:-1, 1:-1] = ((psi_f[2:, 1:-1] - 2*psi_f[1:-1, 1:-1] + psi_f[:-2, 1:-1])/dr**2 + (1.0/R[1:-1, 1:-1])*(psi_f[2:, 1:-1] - psi_f[:-2, 1:-1])/(2*dr) + (psi_f[1:-1, 2:] - 2*psi_f[1:-1, 1:-1] + psi_f[1:-1, :-2])/dz**2)
E_kin_f_SI = -0.5 * (1.0 / mass_f_scaled) * 2 * np.pi * np.sum(np.conj(psi_f) * lap_f * R * dr * dz) * osc_E
total_ground_E_nK = np.real(E_pot_SI + E_kin_b_SI + E_kin_f_SI) / (kB * 1e-9)

print(f"✨ Ground State Solver complete! Total Energy Base: {total_ground_E_nK:.4f} nK")

dt_max_phase_ms = ((0.15 * hbar) / (max(np.max(U_b), np.max(U_f)) * osc_E)) * 1e3
dt_cfl_ms = ((2 * m_f_kg * (min(delta_r, delta_z) * 1e-6)**2) / (np.pi * hbar)) * 1e3
absolute_safe_dt_ms = min(dt_max_phase_ms, dt_cfl_ms)

# Generate Equilibrium Figure
rb_p, rf_p = np.abs(psi_b)**2 * density_conversion, np.abs(psi_f)**2 * density_conversion
fig, ax = plt.subplots(1, 2, figsize=(11, 3))
ax[0].plot(z_vec*osc_r*1e6, np.mean(rb_p[0:3, :], axis=0), 'b', label='Boson')
ax[0].plot(z_vec*osc_r*1e6, np.mean(rf_p[0:3, :], axis=0), 'r', label='Fermion')
ax[0].set_title("Equilibrium State (Axial Core Average)"); ax[0].set_xlabel("Z (μm)"); ax[0].set_ylabel("Density (1/μm³)"); ax[0].legend(); ax[0].grid(True)
ax[1].plot(r_vec*osc_r*1e6, rb_p[:, mid_z_idx], 'b'); ax[1].plot(r_vec*osc_r*1e6, rf_p[:, mid_z_idx], 'r')
ax[1].set_title("Equilibrium State (Radial Slice)"); ax[1].set_xlabel("R (μm)"); ax[1].grid(True)
plt.show()


# =====================================================================
# MODULE 3: USER-CONTROLLED QUENCH DYNAMICS LOOP (STABILIZED OVERRIDE)
# =====================================================================
print("\n" + "="*60)
print("🌀 INITIALIZING MODULE 3: REAL-TIME OVERRIDE ENGINE & EVOLUTION")
print("="*60)

# CRITICAL RESOLUTION LOCK: Set quench grids equal to module 2 resolution to secure matching indices
delta_r_quench = delta_r
delta_z_quench = delta_z
dt_quench_ms = 0.5            # Finer visualization windows (ms)
total_time_ms = 3.0
a_bf_new = -300.0

# Inherit identical scale parameters directly to guarantee continuity
dr_q, dz_q = dr, dz
Nr_q, Nz_q = Nr, Nz
R_q, Z_q = R, Z
r_qvec, z_qvec = r_vec, z_vec
dt_step = (dt_quench_ms * 1e-3) / osc_t

# Assign state variables directly, skipping interpolation routines entirely
psi_b_live = psi_b.copy().astype(complex)
psi_f_live = psi_f.copy().astype(complex)

def norm_q(psi):
    return 2 * np.pi * np.sum(np.abs(psi)**2 * R_q * dr_q * dz_q)

V_b_q = 0.5 * (R_q**2 + (omega_zb_scaled * Z_q)**2)
V_f_q = 0.5 * mass_f_scaled * ((omega_rf_scaled * R_q)**2 + (omega_zf_scaled * Z_q)**2)
g_bf_q = 2 * np.pi * (a_bf_new * 5.2917721e-11) * (m_b_kg / m_R) / osc_r

sponge_q = sponge.copy()

# CRITICAL MULTIPLIER UPDATE: Increase microstep resolution (* 6) to hold the phase-front
# together when density contractions peak past 2.0 ms.
dt_cfl_local_ms = ((2 * m_f_kg * (min(delta_r_quench, delta_z_quench) * 1e-6)**2) / (np.pi * hbar)) * 1e3
sub_divisions = int(np.ceil(dt_quench_ms / min(absolute_safe_dt_ms, dt_cfl_local_ms))) * 6
dt_sub = dt_step / sub_divisions

print(f"⚙️ CFL Adaptive Safety: High-density sub-stepping active = {sub_divisions} loops per window step.\n")

num_intervals = int(np.round((total_time_ms * 1e-3 / osc_t) / dt_step))
mid_zq_idx = int(Nz_q / 2)

for step_idx in range(num_intervals + 1):
    curr_ms = step_idx * dt_quench_ms

    # Calculate ongoing values across conservation benchmarks
    Nb_live = norm_q(psi_b_live)
    Nf_live = norm_q(psi_f_live)

    rb_l, rf_l = np.abs(psi_b_live)**2, np.abs(psi_f_live)**2
    E_pot = np.sum((V_b_q*rb_l + V_f_q*rf_l + 0.5*g_bb*rb_l**2 + g_bf_q*rb_l*rf_l + 0.6*A_fermi*np.maximum(rf_l, 1e-15)**(5/3)) * R_q)
    E_pot_SI = E_pot * 2 * np.pi * dr_q * dz_q * osc_E

    lap_b = np.zeros_like(psi_b_live)
    lap_b[1:-1, 1:-1] = ((psi_b_live[2:, 1:-1] - 2*psi_b_live[1:-1, 1:-1] + psi_b_live[:-2, 1:-1])/dr_q**2 + (1.0/R_q[1:-1, 1:-1])*(psi_b_live[2:, 1:-1] - psi_b_live[:-2, 1:-1])/(2*dr_q) + (psi_b_live[1:-1, 2:] - 2*psi_b_live[1:-1, 1:-1] + psi_b_live[1:-1, :-2])/dz_q**2)
    E_kin_b_SI = -0.5 * 2 * np.pi * np.sum(np.conj(psi_b_live) * lap_b * R_q * dr_q * dz_q) * osc_E

    lap_f = np.zeros_like(psi_f_live)
    lap_f[1:-1, 1:-1] = ((psi_f_live[2:, 1:-1] - 2*psi_f_live[1:-1, 1:-1] + psi_f_live[:-2, 1:-1])/dr_q**2 + (1.0/R_q[1:-1, 1:-1])*(psi_f_live[2:, 1:-1] - psi_f_live[:-2, 1:-1])/(2*dr_q) + (psi_f_live[1:-1, 2:] - 2*psi_f_live[1:-1, 1:-1] + psi_f_live[1:-1, :-2])/dz_q**2)
    E_kin_f_SI = -0.5 * (1.0 / mass_f_scaled) * 2 * np.pi * np.sum(np.conj(psi_f_live) * lap_f * R_q * dr_q * dz_q) * osc_E
    Eq_live = np.real(E_pot_SI + E_kin_b_SI + E_kin_f_SI) / (kB * 1e-9)

    print(f"🕒 Time: {curr_ms:.2f} ms | Energy: {Eq_live:.3f} nK | Bosons: {Nb_live:.1f} | Fermions: {Nf_live:.1f}")

    # Real-Time Snapshot Rendering Graphs Slices
    rho_bq = rb_l * density_conversion
    rho_fq = rf_l * density_conversion

    # Core row average projection for axial profiles
    axial_b_profile = np.mean(rho_bq[0:3, :], axis=0)
    axial_f_profile = np.mean(rho_fq[0:3, :], axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 2.6))
    ax[0].plot(z_qvec*osc_r*1e6, axial_b_profile, 'b', label='Boson')
    ax[0].plot(z_qvec*osc_r*1e6, axial_f_profile, 'r', label='Fermion')
    ax[0].set_title(f"Axial Core Profile | t = {curr_ms:.2f} ms")
    ax[0].set_ylabel("Density (1/μm³)")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(r_qvec*osc_r*1e6, rho_bq[:, mid_zq_idx], 'b')
    ax[1].plot(r_qvec*osc_r*1e6, rho_fq[:, mid_zq_idx], 'r')
    ax[1].set_title(f"Radial Profile | t = {curr_ms:.2f} ms")
    ax[1].set_xlabel("R (μm)")
    ax[1].grid(True)
    plt.tight_layout()
    plt.show()

    if step_idx == num_intervals:
        break

    # Micro-Step Propagation Split Execution
    for sub_step in range(sub_divisions):
        rho_b_loop, rho_f_loop = np.abs(psi_b_live)**2, np.abs(psi_f_live)**2
        mu_fermi_term = A_fermi * np.maximum(rho_f_loop, 1e-15)**(2/3)

        U_b_live = V_b_q + g_bb * rho_b_loop + g_bf_q * rho_f_loop
        U_f_live = V_f_q + mu_fermi_term + g_bf_q * rho_b_loop

        psi_b_live *= np.exp(-1j * U_b_live * dt_sub)
        psi_f_live *= np.exp(-1j * U_f_live * dt_sub)

        for psi, mass_c in [(psi_b_live, 1.0), (psi_f_live, mass_f_scaled)]:
            lap = np.zeros_like(psi)
            lap[1:-1, 1:-1] = (
                (psi[2:, 1:-1] - 2*psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dr_q**2 +
                (1.0 / R_q[1:-1, 1:-1]) * (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2*dr_q) +
                (psi[1:-1, 2:] - 2*psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dz_q**2
            )
            psi += 1j * (0.5 / mass_c) * lap * dt_sub

        psi_b_live *= sponge_q; psi_f_live *= sponge_q

print("\n Simulation complete on custom stabilized user-defined spatial quench constraints.")
