# LLM Regression Model Selection Agent

Compares an LLM-guided model selection agent against GridSearchCV and Optuna on five regression benchmarks. Full write-up in `blog_post.md`.

## Setup

```bash
pip install anthropic scikit-learn pandas numpy optuna python-dotenv
```

Create a `.env` file with your Anthropic API key:

```
ANTHROPIC_API_KEY=your_key_here
```

## Prepare datasets

```bash
python make_datasets.py
```

Generates `data/california.csv`, `data/diabetes.csv`, `data/abalone.csv`, `data/energy.csv`, `data/kin8nm.csv`.

Verify they are valid:

```bash
python test_datasets.py
```

## Run the agent

```bash
python agent.py --data data/energy.csv --target Y1
```

Add `--verbose` to stream the agent's reasoning token-by-token. The report is saved to `results/` automatically.

Available datasets and their targets:

| Dataset | `--data` | `--target` |
|---------|----------|------------|
| California Housing | `data/california.csv` | `median_house_value` |
| Diabetes | `data/diabetes.csv` | `target` |
| Abalone | `data/abalone.csv` | `Rings` |
| Energy Efficiency | `data/energy.csv` | `Y1` |
| Kin8nm | `data/kin8nm.csv` | `y` |

## Run baselines without the agent

Runs GridSearchCV and/or Optuna on all datasets — no API calls, no tokens spent.

```bash
python run_baselines.py                        # both, 50 Optuna trials
python run_baselines.py --optuna --trials 100  # Optuna only, 100 trials
python run_baselines.py --gridsearch           # GridSearchCV only
python run_baselines.py --dataset energy       # single dataset
```
