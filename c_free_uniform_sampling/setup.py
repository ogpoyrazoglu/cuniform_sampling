from setuptools import setup, find_packages

# Mirrors the source repository's editable-install convention so that the
# top-level packages (`classes.*`, `flow_Cuniform.*`, `map_conditioning.*`)
# remain importable exactly as in the original research repo. find_packages()
# discovers every directory containing an __init__.py, preserving the
# absolute import paths the migrated code relies on
# (e.g. `from classes.grid import Grid`,
#  `from flow_Cuniform.dynamics_helpers import ...`).
setup(
    name='traj_sampling',
    version='0.0.0',
    packages=find_packages(),
)
