import sys
import os

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.wing_solver import solve_wing

sections = [
    {"y":0, "chord":1.5, "twist":0, "m":0.02, "p":0.4, "t":0.12},
    {"y":2, "chord":1.2, "twist":-1, "m":0.02, "p":0.4, "t":0.12},
]

res = solve_wing(sections, 6, 5, 40)

print(res)
