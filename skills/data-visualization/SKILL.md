# Data Visualization Skill

## Purpose

Turn a data or visualization request into a reproducible Python visualization workflow that runs in this repository's Docker environment.

## Default stack

Read `docs/library-guide.md` before choosing libraries. Prefer the existing stack unless the task genuinely needs another dependency:

- Streamlit for interactive apps and dashboards
- Pandas for tabular data
- NumPy for numeric work and generated data
- Matplotlib for static/publication-style figures
- Plotly for interactive charts
- scikit-learn for simple modeling and metrics

## Workflow

### 1. Understand the data

Identify:
- source and schema
- dimensions and measures
- missing values and types
- whether values are measured, simulated, or derived

Do not invent real-world data unless the user explicitly asks for synthetic/demo data.

### 2. Choose the visual encoding

Decide what relationship must be communicated first, then choose the chart. Avoid using a chart merely because a library supports it.

### 3. Build a minimal working version

Keep data processing separate from rendering where practical. Use deterministic seeds for synthetic examples.

### 4. Validate

Check:
- axes and units
- labels and legends
- sorting and aggregation
- empty/filter edge cases
- NaN/Inf handling
- whether visual emphasis matches the data rather than distorting it

### 5. Run in Docker/GitHub Actions

Code intended for this repository should run in the Docker environment described in `docs/docker-environment.md`.

Do not assume packages installed on a local machine.

### 6. Review output

For static images, inspect the produced artifact when possible. For Streamlit or Plotly apps, verify startup/imports and preserve a runnable entry point.

## Dependency policy

If a new third-party package is needed, update all four places:

1. `requirements.txt`
2. `docs/library-guide.md`
3. `examples/import_smoke_test.py`
4. Docker/Actions configuration if system packages are required

Never silently add ad-hoc `pip install` commands inside application code.

## Output conventions

- reusable code goes under `src/` or a clearly named app/script
- demos go under `examples/`
- generated output should go under `output/`
- GitHub Actions should upload useful generated artifacts when the task is non-interactive
