import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ===== PD (ligand-receptor) parameters from Task 1 =====
Kd1 = 0.17      # nM, GLP1
Kd2 = 0.077     # nM, Cavalitide (fitted in Task 1)
L1 = 0.015      # nM, GLP1 (fasting)

def Fact(L1, L2, Kd1, Kd2):
    """Fractional receptor activation with two agonists L1 and L2"""
    return (Kd2*L1 + Kd1*L2) / (Kd1*Kd2 + Kd2*L1 + Kd1*L2)

# ===== choose product: 'Ozempic' or 'Wegovy' =====
drug = 'Ozempic'
#drug = 'Wegovy'

# product-specific parameters
products = {
    'Cavalitide 21 mg': dict(Favailable=0.01,  # oral bioavailability
                             ka=0.006,         # 1/h, absorption rate
                             D=21,             # mg per dose
                             interval=24,      # h (daily)
                             n_doses=100),     # 100 days
    'Cavalitide 28.5 mg': dict(Favailable=0.01,
                               ka=0.006,
                               D=28.5,         # mg  <-- changed from 21 mg
                               interval=24,
                               n_doses=100),
}

# shared model parameters
ke = 0.0039     # 1/h, elimination rate
V = 10.8        # L
MW = 4113.6     # g/mol
Ffree = 0.01    # rest of semaglutide bound to albumin

# simulation parameters
dt = 0.1        # h, output time spacing (solve_ivp picks its own internal steps)

def ODEfunc(t, y, ka):
    Asc, Cc = y
    dAsc = -ka*Asc            # Asc: depot amount (mg)
    dCc = ka*Asc/V - ke*Cc    # Cc: blood concentration (mg/L)
    return [dAsc, dCc]

def simulate(Favailable, ka, D, n_doses, interval, dt):
    """Repeated dosing every `interval` h; returns time (h), depot amount (mg),
    and blood concentration (nmol/L)."""
    t_out, Asc_out, Cc_out = [], [], []
    y = np.array([0.0, 0.0])
    n_steps = int(round(interval / dt))
    for i in range(n_doses):
        y[0] += Favailable*D                 # absorbed fraction of dose enters the depot
        t0 = i * interval
        t_seg = t0 + dt * np.arange(n_steps + 1)

        # solve_ivp replaces the explicit Euler loop
        sol = solve_ivp(ODEfunc, (t_seg[0], t_seg[-1]), y, t_eval=t_seg,
                        args=(ka,), method='RK45', rtol=1e-8, atol=1e-10)
        y_seg = sol.y

        t_out.append(t_seg)
        Asc_out.append(y_seg[0])
        Cc_out.append(y_seg[1])
        y = y_seg[:, -1].copy()         # carry state into next interval
    t = np.concatenate(t_out)
    Asc_out = np.concatenate(Asc_out)
    Cc_out = np.concatenate(Cc_out)
    Cc_nM = Cc_out * 1e-3 / MW * 1e9
    return t, Asc_out, Cc_nM

# --- run ---
results = {}
for name, p in products.items():
    t, Asc, Cc = simulate(p['Favailable'], p['ka'], p['D'], p['n_doses'], p['interval'], dt)
    L2_free = Ffree * Cc                      # free Cavalitide (nM) binds the receptor
    F = Fact(L1, L2_free, Kd1, Kd2)           # PD: fraction of active GLP1R
    results[name] = (t, L2_free, F)
    print(f"{name:20s} day 100: free Cavalitide = {L2_free[-1]:.3f} nM, Fact = {F[-1]:.3f}")

print(f"Baseline (GLP1 only, no drug): Fact = {Fact(L1, 0, Kd1, Kd2):.3f}")

# ===== Figure 1 (Task 3.1): PK and PD over 100 days at 21 mg =====
t, L2_free, F = results['Cavalitide 21 mg']
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
ax1.plot(t / 24, L2_free)
ax1.set_ylabel("Free Cavalitide (nM)")
ax1.set_title("Cavalitide 21 mg daily pill: PK and PD")
ax1.grid(alpha=0.3)
ax2.plot(t / 24, F, color='tab:red')
ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Fraction active GLP1R (Fact)")
ax2.set_ylim(0, 1)
ax2.grid(alpha=0.3)
plt.tight_layout()

# ===== Figure 2 (Task 3.2): optimized dose for ~90% active receptors =====
plt.figure()
for name in products:
    t, L2_free, F = results[name]
    plt.plot(t / 24, F, label=f"{name} daily")
plt.axhline(0.9, color='k', ls='--', lw=1, label="target 90% active")
plt.xlabel("Time (days)")
plt.ylabel("Fraction active GLP1R (Fact)")
plt.title("Optimizing Cavalitide dose for receptor activation")
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.show()