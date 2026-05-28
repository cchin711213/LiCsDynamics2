import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_index = "Quantum Mixture App"
st.title("🌌 Bose-Fermi Mixture Quench Dynamics")
st.markdown("Simulating a mixture of Bosonic $^{132}\text{Cs}$ and Fermionic $^6\text{Li}$.")

# =====================================================================
# SIDEBAR CONTROLS (STREAMLIT UI)
# =====================================================================
st.sidebar.header("🔧 System Parameter Configuration")

# Module 1 & 2 Inputs
N_b = st.sidebar.slider("Boson Particle Count (N_b)", 5000, 20000, 10000, step=1000)
N_f = st.sidebar.slider("Fermion Particle Count (N_f)", 10000, 100000, 50000, step=5000)

# Module 3 Overrides
st.sidebar.header("🌀 Real-Time Quench Controls")
a_bf_new = st.sidebar.slider("Post-Quench Interaction a_bf (Bohr)", -400.0, -50.0, -200.0, step=50.0)
total_time_ms = st.sidebar.slider("Simulation Duration (ms)", 1.0, 4.0, 2.5, step=0.5)

# =====================================================================
# MODULE 1 & 2: CACHED EQUILIBRIUM SOLVER
# =====================================================================
@st.cache_data
def run_ground_state(N_b, N_f):
    """
    Runs the entire ground state relaxation sequence once.
    Streamlit caches this so moving quench sliders won't re-trigger it.
    """
    # --- PHYSICAL CONSTANTS & CORE SCALES ---
    m_b_amu = 132.90545 
    m_f_amu = 6.015122  
    nu_rb, nu_zb = 80.0, 6.0          
    nu_rf, nu_zf = 320.0, 24.0         
    a_bb = 280.0         
    a_bf_initial = 0.0            

    hbar = 1.0545718e-34
    u_atomic_unit = 1.66053906e-27
    kB = 1.380649e-23

    m_b_kg = m_b_amu * u_atomic_unit  
    m_f_kg = m_f_amu * u_atomic_unit  

    omega_rb = 2 * np.pi * nu_rb
    osc_r = np.sqrt(hbar / (m_b_kg * omega_rb)) 
    osc_t = 1.0 / omega_rb                      
    osc_E = hbar * omega_rb                     
    density_conversion = 1.0 / ((osc_r * 1e6)**3)

    mass_f_scaled = m_f_kg / m_b_kg
    omega_zb_scaled = (2 * np.pi * nu_zb) / omega_rb
    omega_rf_scaled = (2 * np.pi * nu_rf) / omega_rb
    omega_zf_scaled = (2 * np.pi * nu_zf) / omega_rb

    g_bb = 4 * np.pi * (a_bb * 5.2917721e-11) / osc_r
    m_R = (m_b_kg * m_f_kg) / (m_b_kg + m_f_kg)
    g_bf_initial = 2 * np.pi * (a_bf_initial * 5.2917721e-11) * (m_b_kg / m_R) / osc_r
    A_fermi = 0.5 * (1.0 / mass_f_scaled) * (3 * np.pi**2)**(2/3)

    # --- GRID CONSTANTS ---
    Lr_fixed, Lz_fixed = 25.0, 250.0   
    delta_r, delta_z = 0.35, 1.0       # Balanced for cloud speed & stability
    dt_ground_ms = 0.002 
    max_iterations = 1200

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

    sponge = np.ones((Nr, Nz))
    br, bz = 0.94 * Lr_scaled, 0.94 * Lz_scaled
    sponge[R > br] *= np.exp(-40.0 * ((R[R > br] - br) / (Lr_scaled - br))**4)
    sponge[np.abs(Z) > bz] *= np.exp(-40.0 * ((np.abs(Z[np.abs(Z) > bz]) - bz) / (Lz_scaled - bz))**4)

    def norm_g(psi):
        return 2 * np.pi * np.sum(np.abs(psi)**2 * R * dr * dz)

    # Wavefunction Allocation
    psi_b = np.exp(-(R**2 / 120.0 + Z**2 / 1200.0))
    psi_f = np.exp(-(R**2 / 140.0 + Z**2 / 1400.0))
    psi_b *= np.sqrt(N_b / norm_g(psi_b))
    psi_f *= np.sqrt(N_f / norm_g(psi_f))

    # Relaxation Execution Loop
    for iteration in range(max_iterations):
        rho_b, rho_f = np.abs(psi_b)**2, np.abs(psi_f)**2
        ramp = 1.0 if iteration > 150 else (iteration / 150.0)
        mu_fermi_term = A_fermi * np.maximum(rho_f, 1e-15)**(2/3)
        
        U_b = V_b + (g_bb * rho_b + g_bf_initial * rho_f) * ramp
        U_f = V_f + mu_fermi_term + (g_bf_initial * rho_b) * ramp
        
        psi_b *= np.exp(-np.clip(U_b, -45/dt_g, 45/dt_g) * dt_g)
        psi_f *= np.exp(-np.clip(U_f, -45/dt_g, 45/dt_g) * dt_g)
        
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

    # Package scalars and vectors into a dictionary to pass cleanly to module 3
    results = {
        "psi_b": psi_b, "psi_f": psi_f, "r_vec": r_vec, "z_vec": z_vec,
        "dr": dr, "dz": dz, "R": R, "Z": Z, "sponge": sponge,
        "osc_r": osc_r, "osc_t": osc_t, "osc_E": osc_E, "kB": kB,
        "density_conversion": density_conversion, "mass_f_scaled": mass_f_scaled,
        "omega_zb_scaled": omega_zb_scaled, "omega_rf_scaled": omega_rf_scaled,
        "omega_zf_scaled": omega_zf_scaled, "g_bb": g_bb, "A_fermi": A_fermi,
        "m_b_kg": m_b_kg, "m_f_kg": m_f_kg, "m_R": m_R, "hbar": hbar
    }
    return results

# Fire cached equilibrium step
with st.spinner("Calculating system equilibrium configuration..."):
    sim = run_ground_state(N_b, N_f)

# Display Static Initial State
st.subheader("📊 Initial Equilibrium Profile")
rb_p = np.abs(sim["psi_b"])**2 * sim["density_conversion"]
rf_p = np.abs(sim["psi_f"])**2 * sim["density_conversion"]

fig_init, ax_init = plt.subplots(1, 2, figsize=(10, 3))
ax_init[0].plot(sim["z_vec"]*sim["osc_r"]*1e6, np.mean(rb_p[0:3, :], axis=0), 'b', label='Boson')
ax_init[0].plot(sim["z_vec"]*sim["osc_r"]*1e6, np.mean(rf_p[0:3, :], axis=0), 'r', label='Fermion')
ax_init[0].set_title("Axial Slice"); ax_init[0].set_ylabel("Density (1/μm³)"); ax_init[0].grid(True); ax_init[0].legend()
ax_init[1].plot(sim["r_vec"]*sim["osc_r"]*1e6, rb_p[:, int(len(sim["z_vec"])/2)], 'b')
ax_init[1].plot(sim["r_vec"]*sim["osc_r"]*1e6, rf_p[:, int(len(sim["z_vec"])/2)], 'r')
ax_init[1].set_title("Radial Slice"); ax_init[1].set_xlabel("R (μm)"); ax_init[1].grid(True)
st.pyplot(fig_init)

# =====================================================================
# MODULE 3: REAL-TIME DYNAMICS ENGINE (ST CHROMIUM COMPATIBLE)
# =====================================================================
st.subheader("🌀 Real-Time Evolution Track")
run_dynamics = st.button("🚀 Trigger Real-Time Quench Evolution")

if run_dynamics:
    # Setup Live Metric Display Cards
    metric_time = st.empty()
    metric_energy = st.empty()
    
    # Create an empty visual container that we will continuously update
    plot_placeholder = st.empty()
    
    # Extract variables from state payload dictionary
    psi_b_live = sim["psi_b"].copy().astype(complex)
    psi_f_live = sim["psi_f"].copy().astype(complex)
    R_q, Z_q = sim["R"], sim["Z"]
    dr_q, dz_q = sim["dr"], sim["dz"]
    
    dt_quench_ms = 0.5
    dt_step = (dt_quench_ms * 1e-3) / sim["osc_t"]
    num_intervals = int(np.round((total_time_ms * 1e-3 / sim["osc_t"]) / dt_step))
    mid_zq_idx = int(len(sim["z_vec"]) / 2)
    
    V_b_q = 0.5 * (R_q**2 + (sim["omega_zb_scaled"] * Z_q)**2)
    V_f_q = 0.5 * sim["mass_f_scaled"] * ((sim["omega_rf_scaled"] * R_q)**2 + (sim["omega_zf_scaled"] * Z_q)**2)
    g_bf_q = 2 * np.pi * (a_bf_new * 5.2917721e-11) * (sim["m_b_kg"] / sim["m_R"]) / sim["osc_r"]
    
    # Phase stability sub-division limiter
    sub_divisions = 15 # Standard fast adaptive default slice step count
    dt_sub = dt_step / sub_divisions

    def norm_q(psi):
        return 2 * np.pi * np.sum(np.abs(psi)**2 * R_q * dr_q * dz_q)

    # Time Propagation Loop
    for step_idx in range(num_intervals + 1):
        curr_ms = step_idx * dt_quench_ms
        
        # Energy Diagnostics Integration
        rb_l, rf_l = np.abs(psi_b_live)**2, np.abs(psi_f_live)**2
        E_pot = np.sum((V_b_q*rb_l + V_f_q*rf_l + 0.5*sim["g_bb"]*rb_l**2 + g_bf_q*rb_l*rf_l + 0.6*sim["A_fermi"]*np.maximum(rf_l, 1e-15)**(5/3)) * R_q)
        E_pot_SI = E_pot * 2 * np.pi * dr_q * dz_q * sim["osc_E"]
        
        lap_b = np.zeros_like(psi_b_live)
        lap_b[1:-1, 1:-1] = ((psi_b_live[2:, 1:-1] - 2*psi_b_live[1:-1, 1:-1] + psi_b_live[:-2, 1:-1])/dr_q**2 + (1.0/R_q[1:-1, 1:-1])*(psi_b_live[2:, 1:-1] - psi_b_live[:-2, 1:-1])/(2*dr_q) + (psi_b_live[1:-1, 2:] - 2*psi_b_live[1:-1, 1:-1] + psi_b_live[1:-1, :-2])/dz_q**2)
        E_kin_b_SI = -0.5 * 2 * np.pi * np.sum(np.conj(psi_b_live) * lap_b * R_q * dr_q * dz_q) * sim["osc_E"]
        Eq_live = np.real(E_pot_SI + E_kin_b_SI) / (sim["kB"] * 1e-9)
        
        # Update UI Data Readouts
        metric_time.markdown(f"**Physical Timeline:** `{curr_ms:.2f} ms` / `{total_time_ms:.2f} ms`")
        metric_energy.markdown(f"**Total System Energy:** `{Eq_live:.2f} nK`")
        
        # Graph Slicing Generation
        rho_bq = rb_l * sim["density_conversion"]
        rho_fq = rf_l * sim["density_conversion"]
        axial_b = np.mean(rho_bq[0:3, :], axis=0)
        axial_f = np.mean(rho_fq[0:3, :], axis=0)
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 2.8))
        ax[0].plot(sim["z_vec"]*sim["osc_r"]*1e6, axial_b, 'b', label='Boson')
        ax[0].plot(sim["z_vec"]*sim["osc_r"]*1e6, axial_f, 'r', label='Fermion')
        ax[0].set_ylim(0, np.max(rho_bq)*1.1 + 1.0)
        ax[0].set_title("Axial Density Wave Profiles"); ax[0].set_ylabel("Density (1/μm³)"); ax[0].grid(True); ax[0].legend()
        
        ax[1].plot(sim["r_vec"]*sim["osc_r"]*1e6, rho_bq[:, mid_zq_idx], 'b')
        ax[1].plot(sim["r_vec"]*sim["osc_r"]*1e6, rho_fq[:, mid_zq_idx], 'r')
        ax[1].set_ylim(0, np.max(rho_bq)*1.1 + 1.0)
        ax[1].set_title("Radial Density Profiles"); ax[1].set_xlabel("R (μm)"); ax[1].grid(True)
        
        # Push the generated image canvas directly into the UI placeholder container
        plt.tight_layout()
        plot_placeholder.pyplot(fig)
        plt.close(fig)
        
        if step_idx == num_intervals:
            break
            
        # Microstep Real-Time Sub-Iteration Updates
        for sub in range(sub_divisions):
            rho_b_loop, rho_f_loop = np.abs(psi_b_live)**2, np.abs(psi_f_live)**2
            mu_fermi_term = sim["A_fermi"] * np.maximum(rho_f_loop, 1e-15)**(2/3)
            
            U_b_live = V_b_q + sim["g_bb"] * rho_b_loop + g_bf_q * rho_f_loop
            U_f_live = V_f_q + mu_fermi_term + g_bf_q * rho_b_loop
            
            psi_b_live *= np.exp(-1j * U_b_live * dt_sub)
            psi_f_live *= np.exp(-1j * U_f_live * dt_sub)
            
            for psi, mass_c in [(psi_b_live, 1.0), (psi_f_live, sim["mass_f_scaled"])]:
                lap = np.zeros_like(psi)
                lap[1:-1, 1:-1] = (
                    (psi[2:, 1:-1] - 2*psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dr_q**2 + 
                    (1.0 / R_q[1:-1, 1:-1]) * (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2*dr_q) + 
                    (psi[1:-1, 2:] - 2*psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dz_q**2
                )
                psi += 1j * (0.5 / mass_c) * lap * dt_sub
                
            psi_b_live *= sim["sponge"]; psi_f_live *= sim["sponge"]
            
    st.success("✨ Physical dynamic simulation complete!")
