import sys
from importlib.metadata import version

print("Python", sys.version.split()[0])

packages = [
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "pingouin",
    "gensim",
    "krippendorff",
    "matplotlib",
]

for package in packages:
    try:
        print(package, version(package))
    except Exception:
        print(package, "NOT INSTALLED")
