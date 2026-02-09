import numpy as np
import pandas as pd
import joblib
import os


# ================================
# LOAD TRAINED MODELS
# ================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models_2d")

cl_model = joblib.load(os.path.join(MODEL_DIR, "cl_model.pkl"))
cd_model = joblib.load(os.path.join(MODEL_DIR, "cd_model.pkl"))


# ================================
# CONSTANTS (SI UNITS)
# ================================

RHO = 1.225        # kg/m^3
MU  = 1.81e-5      # air viscosity (kg/m-s)
PI  = np.pi


# ================================
# USER INPUT
# ================================
def get_float(prompt, min_val=None, max_val=None):

    while True:
        try:
            val = float(input(prompt))
            if min_val is not None and val < min_val:
                print(f"Value must be greater than {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"Value must be less than {max_val}")
                continue
            return val
        except ValueError:
            print("Invalid input. Please enter a number")
def get_int(prompt, min_val=None, max_val=None):

    while True:
        try:
            val = int(input(prompt))
            if min_val is not None and val < min_val:
                print(f"Value must be greater than {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"Value must be less than {max_val}")
                continue
            return val
        except ValueError:
            print("Invalid input. Please enter a number")
        
def get_user_sections():

    n = get_int("Number of wing sections: ")

    sections = []

    print("\n--- Enter Section Data ---")

    for i in range(n):

        print(f"\nSection {i+1}")

        sec = {}

        sec["y"] = get_float("Spanwise position y (m, from root): ", min_val=0)
        sec["chord"] = get_float("Chord (m): ", min_val=0)
        sec["twist"] = get_float("Twist (deg, +up): ")
        sec["sweep"] = np.radians(get_float("Quarter-chord sweep (deg): "))
        sec["m"] = get_float("NACA camber m: ", min_val=0, max_val= 0.09)
        sec["p"] = get_float("NACA camber pos p: ", min_val=0, max_val=0.9)
        sec["t"] = get_float("NACA thickness t: ", min_val=0, max_val=0.18)

        sections.append(sec)

    return sections


def get_flight_conditions():

    print("\n--- Flight Conditions ---")

    alpha = get_float("Angle of attack (deg): ")
    V = get_float("Velocity (m/s): ", min_val=0)
    b = get_float("Wing span (m): ", min_val=0)

    return alpha, V, b


# ================================
# CORE SOLVER
# ================================

def solve_wing(sections, span, alpha_deg,V, N=60):

    b2 = span / 2

    # Sort by spanwise position
    sections = sorted(sections, key=lambda s: s["y"])

    # Extract section arrays
    y_sec = np.array([s["y"] for s in sections])
    c_sec = np.array([s["chord"] for s in sections])
    t_sec = np.array([s["twist"] for s in sections])
    m_sec = np.array([s["m"] for s in sections])
    p_sec = np.array([s["p"] for s in sections])
    th_sec = np.array([s["t"] for s in sections])
    sweep_sec = np.array([s["sweep"] for s in sections])

    # Spanwise grid
    y = np.linspace(0, b2, N)

    # Interpolation
    chord = np.interp(y, y_sec, c_sec)
    twist = np.interp(y, y_sec, t_sec)
    m = np.interp(y, y_sec, m_sec)
    p = np.interp(y, y_sec, p_sec)
    th = np.interp(y, y_sec, th_sec)
    sweep = np.interp(y, y_sec, sweep_sec)

    # Dynamic pressure
    q = 0.5 * RHO * V**2

    dL = np.zeros(N)
    dD = np.zeros(N)

    # Loop spanwise
    for i in range(N):

        # Effective AoA (deg)
        sweepAngle = sweep[i]
        alpha_eff = (alpha_deg + twist[i]) * np.cos(sweepAngle)

        # Reynolds number
        V_eff = V * np.cos(sweepAngle)
        Re = RHO * V_eff * chord[i] / MU
        # Input for ML
        X = pd.DataFrame([{
            "AoA": alpha_eff,
            "logRe": np.log10(Re),
            "m": m[i],
            "p": p[i],
            "t": th[i]
        }])

        # Predict
        Cl = cl_model.predict(X)[0]
        Cl = Cl * (np.cos(sweepAngle)**2)
        Cd = cd_model.predict(X)[0]

        # Section forces
        dL[i] = q * chord[i] * Cl
        dD[i] = q * chord[i] * Cd


    # Integrate
    L = 2 * np.trapezoid(dL, y)
    Dp = 2 * np.trapezoid(dD, y)

    S = 2 * np.trapezoid(chord, y)

    AR = span**2 / S

    CL = L / (q * S)
    CDp = Dp / (q * S)

    # Induced drag (Oswald efficiency)
    e = 0.85
    CDi = CL**2 / (PI * AR * e * (np.cos(sweepAngle)**2))

    CD = CDp + CDi

    return {
        "S": S,
        "AR": AR,
        "CL": CL,
        "CDp": CDp,
        "CDi": CDi,
        "CD": CD
    }


# ================================
# MAIN (LOOPED)
# ================================

def main():

    print("\n===== ML Wing Solver (Multi-Section) =====\n")

    while True:

        sections = get_user_sections()

        alpha, V, span = get_flight_conditions()

        print("\nSolving...\n")

        results = solve_wing(
            sections,
            span,
            alpha,
            V
        )

        print("====== RESULTS ======")
        print(f"Wing Area S   : {results['S']:.3f} m^2")
        print(f"Aspect Ratio : {results['AR']:.2f}")
        print(f"CL           : {results['CL']:.4f}")
        print(f"CDp          : {results['CDp']:.5f}")
        print(f"CDi          : {results['CDi']:.5f}")
        print(f"CD Total     : {results['CD']:.5f}")
        print("=====================")

        # Ask to run again
        again = input("\nRun another case? (y/n): ").strip().lower()

        if again != "y":
            print("\n Thanks for using Sarvajit Karanth's wing solver! Exiting solver. Goodbye :).\n")
            break


if __name__ == "__main__":
    main()
