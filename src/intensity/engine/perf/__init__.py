"""Engine performance report: latency per stage, ingest and restart times, data volume, capacity.

    make engine-report            # writes data/reports/engine/<run_id>/ and pushes it to the bucket
    python -m intensity.engine.perf --scenario demo --no-push

It never touches live inference:

* It runs in its own process, on a private `Engine` (or, for the transport check, a private service on a
  free 127.0.0.1 port). Nothing is sent to the live engine; `--live-url` only reads its `/metrics` and
  `/health` once.
* The engine code is unchanged. Timings come from wrappers set on the private engine's own instances
  (`probe.py`); the two module functions it times (`combine`, `build_reasons`) are put back afterwards.
* AMQP publishing and reason polish are forced off, so no FeedEvent reaches the front end and no LLM is
  called. The clock lane runs from computed clip durations and never writes `run/feeds_state.json`.
* The process lowers its CPU priority (`perf.nice`), so on a shared host the live engine wins.
"""
