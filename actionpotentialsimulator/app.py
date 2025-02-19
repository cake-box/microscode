import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import odeint


st.set_page_config(
    page_title="Action Potential Simulator",  # Title of the browser tab
    page_icon="🔬",  # From the default emojis list
    layout="wide"  # Sets the app to use wide screen
)
st.markdown("<h1 style='text-align: center;'>Action Potential Simulator (Hodgkin-Huxley Model)</h1>", unsafe_allow_html=True)

# Activation and inactivation functions
def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
def beta_m(V): return 4 * np.exp(-(V + 65) / 18)
def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
def beta_h(V): return 1 / (1 + np.exp(-(V + 35) / 10))
def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
def beta_n(V): return 0.125 * np.exp(-(V + 65) / 80)

# Stimulus: Brief current pulse
def stimulus(t, I_max, t_start, t_end):
    return I_max if t_start <= t <= t_end else 0

# Full HH model ODEs
def dVm_dt(y, t, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_max, t_start, t_end):
    V, m, h, n = y

    I_stim = stimulus(t, I_max, t_start, t_end)

    # Channel dynamics
    dm_dt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dh_dt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dn_dt = alpha_n(V) * (1 - n) - beta_n(V) * n

    # Channel currents
    I_Na = g_Na * (m**3) * h * (V - E_Na)
    I_K = g_K * (n**4) * (V - E_K)
    I_L = g_L * (V - E_L)

    # Membrane potential change
    dV_dt = (I_stim - (I_Na + I_K + I_L)) / C_m

    return [dV_dt, dm_dt, dh_dt, dn_dt]

# Streamlit UI
#st.title('Action Potential Simulator (Hodgkin-Huxley Model)')

col1, col2, col3 = st.columns([1,1,3])


# User input sliders
with col1:
    C_m = st.slider("Membrane Capacitance (uF/cm²)", 0.5, 2.0, 1.0)
    g_Na = st.slider("Na Max Conductance (mS/cm²)", 10.0, 150.0, 120.0)
    g_K = st.slider("K Max Conductance (mS/cm²)", 5.0, 50.0, 36.0)
    g_L = st.slider("Leak Conductance (mS/cm²)", 0.01, 1.0, 0.3)
    E_Na = st.slider("Na Reversal Potential (mV)", 30, 70, 50)
with col2:
    E_K = st.slider("K Reversal Potential (mV)", -90, -60, -77)
    E_L = st.slider("Leak Reversal Potential (mV)", -80.0, -50.0, -54.4)
    I_max = st.slider("Stimulus Intensity (uA/cm²)", 5.0, 50.0, 10.0)
    t_start = st.slider("Stimulus Start Time (ms)", 0, 10, 1)
    t_end = st.slider("Stimulus End Time (ms)", 0, 10, 2)

# Time array
t = np.linspace(0, 50, 1000)  # 50 ms total

# Initial conditions
V0 = -65
m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))
y0 = [V0, m0, h0, n0]

# Solve ODEs
sol = odeint(dVm_dt, y0, t, args=(C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, I_max, t_start, t_end))

V = sol[:, 0]  # Membrane potential
m = sol[:, 1]  # Na activation
h = sol[:, 2]  # Na inactivation
n = sol[:, 3]  # K activation

with col3:

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(t, V, color='black')
    ax1.set_title("Action Potential Over Time")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Membrane Potential (mV)")
    ax1.grid(True)

    st.pyplot(fig1)

    # Plot Gating Variables
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(t, m, label="m (Na activation)", color='blue')
    ax2.plot(t, h, label="h (Na inactivation)", color='orange')
    ax2.plot(t, n, label="n (K activation)", color='red')
    ax2.set_title("Gating Variables Over Time")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Probability")
    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig2)
