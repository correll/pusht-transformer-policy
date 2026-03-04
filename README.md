# PushT Transformer Policy (Gymnasium + PyMunk)

This project trains and evaluates a transformer policy for the PushT manipulation task.

## Reference article

This notebook is based on and inspired by:

- Nikolaus Correll, *Robotic Behavior Cloning I: Auto-regressive Transformers*  
	https://medium.com/correll-lab/robotic-behavior-cloning-i-auto-regressive-transformers-a7be623f4291

### Credit for evaluation code

The autoregressive rollout/evaluation structure in this repository (observation deque,
predict-then-execute horizon loop, reward/coverage scoring, and GIF export) follows the
approach presented in the Medium article above.

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
- `checkpoints/pusht_transformer.pt` — saved model checkpoint
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
- `vis.gif` is generated during evaluation and is included for demo visualization.
