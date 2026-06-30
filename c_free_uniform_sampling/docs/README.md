# Documentation Index

This folder holds per-module usage guides and data-artifact regeneration
instructions for the public C-Free-Uniform repository. It supplements the
top-level `README.md`: the README gives the project overview, environment setup,
and quick-start commands, while these docs cover entry points whose usage is not
obvious from the README and explain how to regenerate the large data/model files
that are intentionally excluded from the repository.

All file paths referenced here are relative to the repository root
(`c_free_uniform_sampling/`) unless an absolute path is shown. Every script and
config referenced below exists in this repository.

## Layout

```
docs/
├── README.md                       # this index
├── data_artifacts.md               # how to regenerate / obtain every excluded artifact
└── modules/
    ├── flow_cuniform.md            # core C-Uniform sampling + coverage analysis
    ├── map_conditioning.md         # dataset generation, training, analysis entry points
    ├── single_frame_analysis.md    # CFU-vs-C-Uniform planning analysis
    └── navigation_experiments.md   # closed-loop navigation framework
```

## Where to start

| You want to... | Read |
|---|---|
| Generate reachability sets and C-Uniform trajectory samples | [modules/flow_cuniform.md](modules/flow_cuniform.md) |
| Build C-Free-Uniform datasets and train the map-conditioned model | [modules/map_conditioning.md](modules/map_conditioning.md) |
| Reproduce the CFU-vs-C-Uniform single-frame success-rate result | [modules/single_frame_analysis.md](modules/single_frame_analysis.md) |
| Run closed-loop navigation experiments | [modules/navigation_experiments.md](modules/navigation_experiments.md) |
| Regenerate a `.pkl` dataset, a `.pth`/`.pt` checkpoint, `costmap.npy`/`sdf.npy`, etc. | [data_artifacts.md](data_artifacts.md) |

## Conventions and caveats

- **Editable install.** Run `pip install -e .` from the repository root once so
  that `classes.*`, `flow_Cuniform.*`, and `map_conditioning.*` import correctly.
- **Working directory matters.** Several scripts write their outputs into the
  current working directory or into a folder next to the script. The per-module
  docs note this per entry point.
- **Hardcoded absolute paths.** A number of scripts still contain absolute paths
  from the original research machine (for example `/home/mikasa/RSN/...`). These
  point at excluded data/model artifacts and must be updated before the affected
  script will run. Each occurrence is listed in [data_artifacts.md](data_artifacts.md)
  with the file, line, and what to change it to.
- **Excluded artifacts.** No large datasets or trained weights ship with this
  repository. Anything a script needs that is not present must be regenerated
  using the instructions in [data_artifacts.md](data_artifacts.md).
