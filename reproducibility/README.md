# Numerical verification materials — v1.0

Paper DOI: https://doi.org/10.5281/zenodo.22377652

The scripts check representative numerical examples; they do not verify all mathematical proofs.

Use Python 3.12, NumPy 2.3.5, and SciPy 1.17.0. From the extracted package root:

```sh
python3 -m pip install numpy==2.3.5 scipy==1.17.0
python3 reproducibility/evidence/scripts/verify_reduced_examples.py
python3 reproducibility/evidence/scripts/verify_joint_witness.py
```

Both scripts contain their own model parameters and require no input data files. Successful runs print JSON with `"status": "PASS"` and exit with code 0. The first script also writes `REDUCED_EXAMPLES.json` and `FIGURE5_REDUCED_POINTS.csv` under `reproducibility/evidence/results/`. The second prints the 51-variable compatibility result, including solver diagnostics and support contrasts. Save the output if needed for comparison.

## Rights

Copyright © 2026 Kai Liang.

To the extent that copyright and related rights subsist in material created or lawfully controlled by the author, all rights are reserved. No licence is granted to reproduce, redistribute, translate, adapt, or commercially use this work, except as permitted by applicable law or with prior written permission. Any identified third-party material remains subject to its respective rights and licence terms.
