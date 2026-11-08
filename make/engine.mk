# Engine agent targets.
.PHONY: engine mock mock-ingest samples eval-imminent sync-sheet sync-check

engine:  ## Run the engine service (OVERLAY=local|demo)
	$(PY) -m intensity.engine.service --overlay $(or $(OVERLAY),local)

mock:  ## Engine on the in-process mock generator: http://127.0.0.1:8000/debug
	$(PY) -m intensity.engine.service --overlay local --mock

mock-ingest:  ## Stream mock events over WebSocket to a running engine (URL=...)
	$(PY) -m intensity.engine.mock --url $(or $(URL),ws://127.0.0.1:8000/ingest)

samples:  ## Regenerate contracts/samples/ ScoreUpdates from the mock
	$(PY) scripts/make_samples.py

eval-imminent:  ## Imminent check: SPORT=football SET=tuning|demo|mock
	@test -n "$(SPORT)" || (echo "usage: make eval-imminent SPORT=<sport> SET=tuning|demo|mock" && exit 2)
	$(PY) -m intensity.training.imminent_eval --sport $(SPORT) --set $(or $(SET),tuning)

sync-sheet:  ## Frames at every key match event: CLIP=<clip_id>
	@test -n "$(CLIP)" || (echo "usage: make sync-sheet CLIP=<clip_id>" && exit 2)
	$(PY) -m intensity.engine.syncsheet --clip $(CLIP)

sync-check:  ## Does match data line up with the video? CLIP=<clip_id>|demo (FROM_BUCKET=1 streams the clip, no local copy)
	@test -n "$(CLIP)" || (echo "usage: make sync-check CLIP=<clip_id>|demo [FROM_BUCKET=1]" && exit 2)
	$(PY) -m intensity.engine.synccheck --clip $(CLIP) $(if $(filter 1,$(FROM_BUCKET)),--from-bucket)
