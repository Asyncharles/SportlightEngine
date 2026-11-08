# Training agent targets.
.PHONY: fetch-% convert-% build-% train-% eval-%

define TODO
	@echo "$@: not implemented yet (owner: $(1) agent, milestone $(2))" && exit 1
endef

fetch-%:  ## Download raw data, resumable: make fetch-football|fetch-f1|fetch-nba|fetch-all (ARGS="--limit 1")
	$(PY) -m intensity.training.cli fetch --sport $* $(ARGS)

convert-%:  ## Raw files to match-record JSONL: make convert-football|convert-f1|convert-nba|convert-all
	$(PY) -m intensity.training.cli convert --sport $* $(ARGS)

build-%:  ## Source + featurize + label one sport (training agent, T1-T4)
	$(call TODO,training,T1-T4)

train-%:  ## Train and calibrate one sport (training agent, T1-T3)
	$(call TODO,training,T1-T3)

eval-%:  ## Evaluate one sport on test (training agent, T1-T3)
	$(call TODO,training,T1-T3)
