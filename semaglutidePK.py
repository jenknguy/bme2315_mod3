import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

# ===== choose product: 'Ozempic' or 'Wegovy' =====
drug = 'Ozempic'
#drug = 'Wegovy'

# product-specific parameters
products = {
    'Ozempic': dict(Favailable=0.89,   # bioavailability
                    ka=0.029,          # 1/h, absorption rate
                    D=2.0,             # mg per dose
                    interval=168,      # h between doses (weekly)
                    n_doses=8,         # number of doses
                    depot='subcutaneous'),
    'Wegovy':  dict(Favailable=0.01,
                    ka=0.0029,
                    D=25,               # mg
                    interval=24,       # h between doses (daily)
                    n_doses=100,
                    depot='gut (oral)'),
    'Cavalitide': dict(Favailable=0.01, # cavalitide oral bioavailability
                       ka=0.006,          # 1/h, increased absorption rate
                       D=25,              # mg
                       interval=24,       # h (daily)
                       n_doses=100,
                       depot='gut (oral)'),
    'Cavalitide (optimized)': dict(Favailable=0.01, # optimized Cavalitide dosing (lower daily dose)
                       ka=0.006,
                       D=21,              # mg  <-- changed from 25 mg
                       interval=24,       # h (daily)
                       n_doses=100,
                       depot='gut (oral)'),
}

p = products[drug]
Favailable, ka, D = p['Favailable'], p['ka'], p['D']
interval, n_doses, depot = p['interval'], p['n_doses'], p['depot']

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

# # simulation parameters
# dt = 0.1        # h, Euler step size

# def ODEfunc(t, y):
#     Asc, Cc = y
#     dAsc = -ka*Asc            # Asc: depot amount (mg)
#     dCc = ka*Asc/V - ke*Cc    # Cc: blood concentration (mg/L)
#     return [dAsc, dCc]

# def simulate(D, n_doses, interval, dt):
#     """Repeated dosing every `interval` h; returns time (h), depot amount (mg),
#     and blood concentration (nmol/L)."""
#     t_out, Asc_out, Cc_out = [], [], []
#     y = np.array([0.0, 0.0])
#     n_steps = int(round(interval / dt))      # Euler steps per dosing interval
#     for i in range(n_doses):
#         y[0] += Favailable*D                 # absorbed fraction of dose enters the depot
#         t0 = i * interval
#         t_seg = t0 + dt * np.arange(n_steps + 1)
#         y_seg = np.zeros((2, n_steps + 1))
#         y_seg[:, 0] = y
        
#         # explicit Euler: y[n+1] = y[n] + dt * f(t[n], y[n])
#         for n in range(n_steps):
#             y_seg[:, n + 1] = y_seg[:, n] + dt * np.array(ODEfunc(t_seg[n], y_seg[:, n]))
            
#         t_out.append(t_seg)
#         Asc_out.append(y_seg[0])
#         Cc_out.append(y_seg[1])
#         y = y_seg[:, -1].copy()         # carry state into next interval
#     t = np.concatenate(t_out)
#     Asc_out = np.concatenate(Asc_out)
#     Cc_out = np.concatenate(Cc_out)
#     Cc_nM = Cc_out * 1e-3 / MW * 1e9
#     return t, Asc_out, Cc_nM

# --- run ---

results = {}

for drug_name, p in products.items():
    t, Asc, Cc = simulate(
        p['Favailable'],
        p['ka'],
        p['D'],
        p['n_doses'],
        p['interval'],
        dt
    )

    results[drug_name] = (t, Asc, Cc)

    last = t >= t[-1] - p['interval']

    print(f"{drug_name:24s} free Cc over last interval: mean {np.mean(Ffree*Cc[last]):.3f} nM "
          f"(min {np.min(Ffree*Cc[last]):.3f}, max {np.max(Ffree*Cc[last]):.3f})")
 
# --- Figure 1: Ozempic vs Wegovy ---
plt.figure()
for drug in ['Ozempic', 'Wegovy']:
    t, Asc, Cc = results[drug]
    plt.plot(t / 24, Ffree*Cc, label=f"{drug}: {products[drug]['D']} mg every {products[drug]['interval']} h")
plt.xlabel("Time (days)")
plt.ylabel("Free semaglutide in blood (nM)")
plt.title("Ozempic (weekly injection) vs Wegovy (daily pill)")
plt.legend()
plt.grid(alpha=0.3)
 
# --- Figure 2: Cavalitide vs Wegovy ---
plt.figure()
for drug in ['Wegovy', 'Cavalitide']:
    t, Asc, Cc = results[drug]
    plt.plot(t / 24, Ffree*Cc, label=f"{drug} (ka = {products[drug]['ka']} 1/h)")
plt.xlabel("Time (days)")
plt.ylabel("Free drug in blood (nM)")
plt.title("Cavalitide daily pill, 25 mg")
plt.legend()
plt.grid(alpha=0.3)
 
# --- Figure 3: optimized Cavalitide dosing ---
plt.figure()
for drug in ['Cavalitide', 'Cavalitide (optimized)']:
    t, Asc, Cc = results[drug]
    plt.plot(t / 24, Ffree*Cc, label=f"{drug}: {products[drug]['D']} mg every {products[drug]['interval']} h")
plt.axhline(0.5, color='k', ls='--', lw=1, label="target 0.5 nM")
plt.xlabel("Time (days)")
plt.ylabel("Free Cavalitide in blood (nM)")
plt.title("Optimized Cavalitide dosing")
plt.legend()
plt.grid(alpha=0.3)
 
plt.show()
