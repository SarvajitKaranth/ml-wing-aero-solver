# ml-wing-aero-solver

A machine-learning-based wing aerodynamic analysis tool that predicts lift and drag
using trained airfoil models and spanwise integration.

This project uses trained regression models to approximate airfoil performance and
compute full-wing aerodynamic coefficients.



##  Overview

This solver estimates the aerodynamic performance of a finite wing by:

- Using ML models trained on generated airfoil data
- Predicting section lift and drag coefficients
- Integrating forces across the span
- Estimating induced drag using lifting-line theory

It provides fast aerodynamic analysis without repeatedly running XFOIL.



##  Features

- Multi-section wing definition
- Machine-learning-based Cl/Cd prediction
- Reynolds number correction
- Automatic spanwise interpolation
- Induced drag estimation
- Command-line interface

##  Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ml-wing-aero-solver.git
cd ml-wing-aero-solver
Install dependencies:

pip install -r requirements.txt

## Machine Learning Models

This project uses trained scikit-learn models:

cl_model.pkl — Lift coefficient predictor

cd_model.pkl — Drag coefficient predictor

## Inputs:
Number of sections
For each section: 
Spanwise position
Chord Length
Twist
NACA parameters (m, p, t) 

After:
Angle of attack
Velocity 
Wing span

m represents the first number in 4 digit naca code. Ranges between 0 and 0.09. (enter 0.02 for the 2 in naca 2412)
p represents the second number in 4 digit naca code. Ranges between 0 and 0.9. (enter 0.4 for the 4 in naca 2412)
t represents the last two numberrs in 4 digit naca code. Ranges between 0 and 0.18. (enter 0.12 for the 12 in naca 2412)


## Training Data

Training data was generated using aerodynamic simulations from XFOIL on a range of different airfoils, reynolds numbers, angles of attack, and processed to build
a regression dataset for airfoil performance prediction.


##  Future Improvements Planned

Viscous/inviscid coupling

3D vortex lattice method

GUI interface

Support for non-NACA airfoils

GPU acceleration

## Author

Sarvajit Karanth

Aerospace Engineering Student @ Purdue Univerrsity
Interested in CFD, Machine Learning, and Aircraft Design