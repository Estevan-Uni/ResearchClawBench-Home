# H0 Local Distance Network – minimal dataset

Run `python code/run_all.py` from the workspace root. It writes outputs/*.json, outputs/*.txt and report/images/*.png.

- `h0dn_core.py`: safe parser (ast.literal_eval), compact GLS system with dense anchor/method-anchor covariance, solver, R22-style clipping, explicit-nuisance cross-check.
- `run_all.py`: sanity arithmetic, headline fits, rung-1/2 host residuals, HF convention audit, variants, leave-one-out, Hubble diagrams, MCMC, mocks, injection tests, error budget and Planck comparison.

Requirements: numpy, scipy, matplotlib, emcee. Seeds are fixed.
