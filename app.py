import os
import sys

import numpy as np
import pandas as pd
import gradio as gr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.wing_solver import solve_wing  # noqa: E402

COLS = ["y (m)", "chord (m)", "twist (deg)", "sweep (deg)", "m", "p", "t"]

EXAMPLE_SECTIONS = pd.DataFrame(
    [
        {"y (m)": 0.0, "chord (m)": 1.5, "twist (deg)": 0.0,
         "sweep (deg)": 0.0, "m": 0.02, "p": 0.4, "t": 0.12},
        {"y (m)": 2.0, "chord (m)": 1.2, "twist (deg)": -1.0,
         "sweep (deg)": 0.0, "m": 0.02, "p": 0.4, "t": 0.12},
        {"y (m)": 3.0, "chord (m)": 0.8, "twist (deg)": -2.0,
         "sweep (deg)": 5.0, "m": 0.02, "p": 0.4, "t": 0.12},
    ]
)


def run_solver(sections_df, alpha, V, span):
    if sections_df is None or len(sections_df) < 2:
        return "Enter at least two wing sections.", None

    sections = []
    for _, row in sections_df.iterrows():
        try:
            sections.append({
                "y": float(row["y (m)"]),
                "chord": float(row["chord (m)"]),
                "twist": float(row["twist (deg)"]),
                "sweep": np.radians(float(row["sweep (deg)"])),
                "m": float(row["m"]),
                "p": float(row["p"]),
                "t": float(row["t"]),
            })
        except (ValueError, TypeError):
            return "Every cell must be a number — check for blank rows.", None

    try:
        results = solve_wing(sections, span, alpha, V)
    except Exception as e:  # surface solver/model errors in the UI, not a 500
        return f"Solver error: {e}", None

    ld = results["CL"] / results["CD"] if results["CD"] else float("nan")
    summary = (
        f"Wing area S   : {results['S']:.3f} m^2\n"
        f"Aspect ratio  : {results['AR']:.2f}\n"
        f"CL            : {results['CL']:.4f}\n"
        f"CDp (profile) : {results['CDp']:.5f}\n"
        f"CDi (induced) : {results['CDi']:.5f}\n"
        f"CD total      : {results['CD']:.5f}\n"
        f"L/D           : {ld:.2f}"
    )

    chart_df = pd.DataFrame({
        "Component": ["CDp", "CDi"],
        "Drag coefficient": [results["CDp"], results["CDi"]],
    })

    return summary, chart_df


with gr.Blocks(title="ML Wing Aerodynamics Solver") as demo:
    gr.Markdown(
        "# ML Wing Aerodynamics Solver\n"
        "Predicts lift and drag for a multi-section finite wing using "
        "ML-trained airfoil models plus spanwise integration. Add one row "
        "per wing section, root to tip."
    )

    sections_input = gr.Dataframe(
        value=EXAMPLE_SECTIONS,
        headers=COLS,
        datatype=["number"] * len(COLS),
        row_count=(2, "dynamic"),
        col_count=(len(COLS), "fixed"),
        label="Wing sections (root -> tip)",
    )

    with gr.Row():
        alpha_input = gr.Number(value=5.0, label="Angle of attack (deg)")
        V_input = gr.Number(value=40.0, label="Velocity (m/s)")
        span_input = gr.Number(value=6.0, label="Wing span (m)")

    solve_btn = gr.Button("Solve", variant="primary")

    with gr.Row():
        summary_output = gr.Textbox(label="Results", lines=8)
        chart_output = gr.BarPlot(
            x="Component", y="Drag coefficient", title="Drag breakdown"
        )

    solve_btn.click(
        run_solver,
        inputs=[sections_input, alpha_input, V_input, span_input],
        outputs=[summary_output, chart_output],
    )

if __name__ == "__main__":
    demo.launch()
