# PushT Transformer Policy (Gymnasium + PyMunk)

This project trains and evaluates a transformer policy for the PushT manipulation task.

## Demo

![PushT rollout demo](vis.gif)

## What it does

- Loads PushT demonstration data from a Zarr replay dataset.
- Trains a transformer to predict short-horizon 2D control actions.
- Runs closed-loop evaluation in a custom `PushTEnv` built with Gymnasium + PyMunk.
- Renders rollouts to RGB frames and exports an animated GIF (`vis.gif`).

## Notebook workflow

Main notebook: `pusht.ipynb`

1. Download and extract the dataset.
2. Build normalized training sequences from demonstrations.
3. Train `PushTTransformer` with teacher forcing.
4. Evaluate the policy autoregressively in `PushTEnv`.
5. Save and display rollout GIFs.

## Environment details

- Observation: `[agent_x, agent_y, block_x, block_y, block_angle]`
- Action: `[target_agent_x, target_agent_y]`
- Physics: PyMunk
- API: Gymnasium
- Rendering: NumPy + scikit-image drawing utilities

## Files

- `pusht.ipynb` — end-to-end notebook (data, model, env, eval)
- `requirements.txt` — pinned local environment
- `requirements-colab.txt` — lightweight Colab setup

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open `pusht.ipynb` and run cells in order.

## Notes

- The repository ignores large dataset artifacts (`*.zarr`, `*.zip`) by default.
- `vis.gif` is generated during evaluation and is not versioned by default.
