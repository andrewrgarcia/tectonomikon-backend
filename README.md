# Tectonomikon Backend

> A structural economic simulation engine. Pick a macro variable, shock it, 
> and watch the causal system propagate — narrative interpretation included, 
> zero API costs, runs entirely on your hardware.

Frontend: [tectonomikon.vercel.app](https://tectonomikon.vercel.app)

---

## What it does

Instead of predicting outputs, Tectonomikon evolves economic state through a 
learned operator:

```
x(t+1) = A x(t)
```

The A matrix is inferred from a curated FRED monthly dataset. Shock any 
variable, simulate forward, and extract the dominant causal paths through 
the system.

An optional local LLM (Phi-3) interprets the output using RAG over system 
state and session memory — so it knows what it just simulated.

---

## Why local-first?

Most AI projects quietly assume an API bill somewhere. This one doesn't.

- LLM runs via local Phi-3 — no OpenAI, no Anthropic, no usage costs
- No data leaves your machine
- The hosted fallback (Render) handles variable selection only
- Users can run the full stack themselves with a single command

The frontend auto-detects if you're running this backend locally and upgrades 
itself silently — faster Bayesian variable selection, LLM narration enabled. 
Otherwise it falls back to the hosted server.

---

## Stack

- FastAPI
- Pandas / NumPy / scikit-learn
- Sentence Transformers (MiniLM) — local embeddings for RAG
- Phi-3-mini-4k-instruct — optional local LLM (4-bit quantized)
- Millipede — optional Bayesian variable selection
- Parquet datasets (FRED monthly, 1994–present)

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/andrewrgarcia/tectonomikon-backend
cd tectonomikon-backend
pip install -r requirements.txt
```

### 2. Run

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 10000 --reload
```

Backend available at `http://localhost:10000`

### 3. Enable local LLM (optional)

```bash
pip install transformers torch bitsandbytes accelerate
export TECTO_LLM=phi3
```

Requires a CUDA-capable GPU. Runs Phi-3-mini in 4-bit quantization.

### 4. Enable Bayesian variable selection (optional)

```bash
pip install git+https://github.com/BasisResearch/millipede.git
```

The system automatically detects and uses Millipede if installed, 
falls back to fast correlation otherwise.

---

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /search` | Search FRED macro series by keyword |
| `POST /build-model` | Select variables + fit A matrix |
| `POST /simulate` | Run shock propagation (24 steps) |
| `POST /ask` | LLM narrative via RAG over system state |
| `GET /capabilities` | Returns active LLM + selector status |

---

## Data

Data files are not included in the repository. Expected at:

```
data/
  fred_monthly_master_1994.parquet
  fred_id_to_title.json
```

On first run, the parquet file is downloaded automatically if not present.
For deployment, external mounting is recommended.

---

## Deployment notes

| Mode | Variable selection | LLM |
|---|---|---|
| Hosted (Render) | Correlation | Disabled |
| Local | Millipede (if installed) | Phi-3 (if installed) |

The system is designed for hybrid execution — remote for availability, 
local for full capability.

---

## License

MIT
