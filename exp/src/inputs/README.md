# Frozen Experiment Inputs

This directory contains the two final-report targets derived from recorded
Cartesian commands and the downstream driver envelope used by the simulated
plant:

| File | Purpose |
|---|---|
| `near_chest_fast_wrist_roll.npz` | Fast wrist roll with simultaneous near-chest translation |
| `fast_retract_elbow_branch.npz` | Fast retract used to compare elbow-branch behavior |
| `experiment_driver_config.yaml` | Frozen eight-axis driver velocity limits; the simulation uses the first seven arm values |

All other report trajectories are deterministic and are generated in memory by
the scenario builders. The complete list, generator name, suite usage, and
trajectory hash are recorded in [`../TRAJECTORIES.md`](../TRAJECTORIES.md).
Regenerate or inspect that catalog with `../trajectory_catalog.py`; it does not
write standalone trajectory arrays.

The frozen trajectory files contain `times`, `phase`, left/right target poses,
and initial joint configurations. Their SHA-256 hashes, together with the
driver configuration hash, are pinned in
[`../../manifests/manifest.json`](../../manifests/manifest.json).
