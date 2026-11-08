# Ingest agent targets. Only the clock lane exists (deploy plan D1); the rest are placeholders until the
# ingest agent lands I1 to I6, P1 to P3 and B1.
.PHONY: feeds-up ingest ingest-clock demo record-tapes video-calibrate transcripts clock-check bench-e3 report

define TODO_INGEST
	@echo "$@: not implemented yet (owner: ingest agent, milestone $(1))" && exit 1
endef

feeds-up:  ## MediaMTX + one looping RTSP pusher per feed (I1)
	$(call TODO_INGEST,I1)
ingest:  ## GPU ingest process (I2-I5)
	$(call TODO_INGEST,I2-I5)
ingest-clock:  ## The clock lane: ClockTicks for every feed with a clip in data/samples, to a local engine (make engine)
	$(PY) -m intensity.ingest.run --overlay $(or $(OVERLAY),local)
demo:  ## Engine + clock lane in Docker: docker/compose.yaml (the VM adds compose.gcp.yaml)
	docker compose -f docker/compose.yaml up --build
record-tapes:  ## Event tapes for SET=tuning|demo (I6)
	$(call TODO_INGEST,I6)
video-calibrate:  ## models/video_v0/calibration.json (P3)
	$(call TODO_INGEST,P3)
transcripts:  ## Offline ASR transcripts (P2)
	$(call TODO_INGEST,P2)
clock-check:  ## Anchor within one frame on the test clip (I2)
	$(call TODO_INGEST,I2)
bench-e3:  ## Capacity experiment E3 (B1)
	$(call TODO_INGEST,B1)
report:  ## E3 report from Parquet: RUN=<id> (B1)
	$(call TODO_INGEST,B1)
