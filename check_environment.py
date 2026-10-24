import sys

import numpy as np
import pandas as pd
import quantstats as qs
import qlib
import finrl
import stable_baselines3 as sb3

print("=" * 50)
print("Python Environment")
print("=" * 50)

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Stable-Baselines3:", sb3.__version__)

print("\n" + "=" * 50)
print("GitHub Projects")
print("=" * 50)

print("FinRL:", finrl.__file__)
print("QuantStats:", qs.__file__)
print("Qlib:", qlib.__file__)

print("\nAll basic imports successful!")