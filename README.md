# Tectonomikon Backend

FastAPI backend for a causal macroeconomic simulation system.

---

## Features

- Variable selection (Bayesian / Millipede-style or fast correlation fallback)
- Shock-based simulation
- Causal path extraction
- Search over FRED macroeconomic series

---

## Endpoints

- `/search` → search macro variables
- `/build-model` → select variables + build system
- `/simulate` → run shock propagation

---

## Stack

- FastAPI
- Pandas / NumPy
- Parquet datasets (FRED)

---

## Local Development (IMPORTANT)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Run backend locally (Uvicorn)

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 10000 --reload
```

Backend will be available at:

```text
http://localhost:10000
```

---

### 3. Use with frontend

If running frontend locally or via Vercel:

* Switch backend to **Local (millipede)** in UI
* Or set:

```env
NEXT_PUBLIC_API_URL=http://localhost:10000
```

---

### 4. Millipede (optional, advanced)

Millipede is **not installed by default** due to deployment constraints.

To enable it locally:

```bash
pip install git+https://github.com/BasisResearch/millipede.git
```

The system will automatically use:

```text
millipede → if available
correlation fallback → otherwise
```

---

## Data

Data files are not included in the repository.

Expected:

```text
data/
  fred_monthly_master_1994.parquet
  fred_id_to_title.json
```

For deployment, data can be:

* downloaded at runtime
* or mounted externally (recommended)

---

## Notes

* Render deployment uses correlation fallback (no millipede)
* Local backend supports full computation (millipede)
* Designed for hybrid execution (remote + local compute)

