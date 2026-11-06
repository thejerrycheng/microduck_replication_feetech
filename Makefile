# Microduck HD1910 replica RL -- Isaac Lab 3.0 + Newton MJWarp
ISAACLAB ?= external/IsaacLab
UV       ?= uv run --project $(ISAACLAB) python
PRESET   ?= presets=backlash
NUM_ENVS ?= 4096
VIDEO    ?= --video --video_length 500 --video_interval 6000   # one clip per 250 iterations (24 steps each)

.PHONY: help setup install list-envs smoke measure train-walk train-run train-sit train-stand train-sitstand play record

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

setup: ## clone Isaac Lab main (Newton) into external/ and create its environment
	test -d $(ISAACLAB) || git clone --branch main https://github.com/isaac-sim/IsaacLab.git $(ISAACLAB)
	cd $(ISAACLAB) && uv sync
	$(MAKE) install

install: ## install this package (editable) into the Isaac Lab environment
	uv pip install -e . --python $(ISAACLAB)/.venv/bin/python

list-envs: ## list the registered Microduck tasks
	$(UV) -c "import microduck_rl, gymnasium as gym; [print(t) for t in microduck_rl.TASK_IDS]"

smoke: ## 64-env, 3-iteration training of every task (catches config errors in minutes)
	for t in Microduck-Velocity-Flat-HD1910 Microduck-Sit-Flat-HD1910 Microduck-Stand-Flat-HD1910; do \
	  $(UV) scripts/train.py --rl_library rsl_rl --task $$t --num_envs 64 --max_iterations 3 $(PRESET) || exit 1; done

measure: ## measure standing/seated rest heights of the Newton model (checks STAND_Z / SIT_Z)
	$(UV) scripts/measure_rest_heights.py $(PRESET) --out docs/rest_heights.json

train-walk: ## velocity tracking, flat
	$(UV) scripts/train.py --rl_library rsl_rl --task Microduck-Velocity-Flat-HD1910 --num_envs $(NUM_ENVS) $(PRESET) $(VIDEO) $(ARGS)
train-run: ## velocity tracking with the fast forward range
	$(UV) scripts/train.py --rl_library rsl_rl --task Microduck-Run-Flat-HD1910 --num_envs $(NUM_ENVS) $(PRESET) $(VIDEO) $(ARGS)
train-sit: ## stand -> sit
	$(UV) scripts/train.py --rl_library rsl_rl --task Microduck-Sit-Flat-HD1910 --num_envs $(NUM_ENVS) $(PRESET) $(VIDEO) $(ARGS)
train-stand: ## sit -> stand
	$(UV) scripts/train.py --rl_library rsl_rl --task Microduck-Stand-Flat-HD1910 --num_envs $(NUM_ENVS) $(PRESET) $(VIDEO) $(ARGS)
train-sitstand: ## commanded sit <-> stand, one policy
	$(UV) scripts/train.py --rl_library rsl_rl --task Microduck-SitStand-Flat-HD1910 --num_envs $(NUM_ENVS) $(PRESET) $(VIDEO) $(ARGS)

play: ## TASK=... CKPT=latest|path -- play, record a clip, export policy.onnx
	$(UV) scripts/play.py --rl_library rsl_rl --task $(TASK) --checkpoint $(CKPT) --num_envs 4 $(PRESET) --video --video_length 600 $(ARGS)

record: ## TASK=... RUN=<run dir> -- one clip per checkpoint into rollouts/
	$(UV) scripts/record_checkpoints.py --task $(TASK) --run $(RUN) --every $(or $(EVERY),1) $(PRESET) $(ARGS)
