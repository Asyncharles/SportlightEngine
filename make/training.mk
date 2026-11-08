# Training agent targets.
.PHONY: fetch-% convert-% build-% train-% eval-% export-%

fetch-%:  ## Download raw data, resumable: make fetch-football|fetch-f1|fetch-nba|fetch-all (ARGS="--limit 1")
	$(PY) -m intensity.training.cli fetch --sport $* $(ARGS)

convert-%:  ## Raw files to match-record JSONL: make convert-football|convert-f1|convert-nba|convert-all
	$(PY) -m intensity.training.cli convert --sport $* $(ARGS)

build-%:  ## Source through export in one command: make build-football (ARGS="--no-source")
	$(PY) -m intensity.training.cli build --sport $* $(ARGS)

train-%:  ## Featurize, label, split, train and calibrate one sport (stages run/training/<sport>/<version>/)
	$(PY) -m intensity.training.cli train --sport $* $(ARGS)

eval-%:  ## Evaluate the latest staged bundle on test: reports/training/<sport>/<version>/report.html
	$(PY) -m intensity.training.cli evaluate --sport $* $(ARGS)

export-%:  ## Copy the evaluated bundle to models/<sport>/<version>/ (CURRENT moves only with the ML lead)
	$(PY) -m intensity.training.cli export --sport $* $(ARGS)
