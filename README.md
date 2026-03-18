# Tectonomikon Backend

FastAPI backend for a causal macroeconomic simulation system.

## Features

- Variable selection (Bayesian / Millipede-style)
- Shock-based simulation
- Causal path extraction
- Search over FRED macroeconomic series

## Endpoints

- `/search` → search macro variables
- `/build-model` → select variables + build system
- `/simulate` → run shock propagation

## Stack

- FastAPI
- Pandas / NumPy
- Parquet datasets (FRED)

## Notes

Data files are not included in the repository.