# Bucket sync (src/intensity/data.py, scripts/sync_bucket.sh). FOLDERS="samples events/nba";
# DELETE=1 removes destination files missing at the source; DRY_RUN=1 only lists.
.PHONY: data-pull data-push data-verify

SYNC_FLAGS = $(if $(filter 1,$(DELETE)),--delete) $(if $(filter 1,$(DRY_RUN)),--dry-run)

data-pull:  ## Pull bucket folders into data/ (FOLDERS=..., DELETE=1, DRY_RUN=1)
	bash scripts/sync_bucket.sh pull $(SYNC_FLAGS) $(FOLDERS)

data-push:  ## Push data/ folders to the bucket (FOLDERS=..., DELETE=1, DRY_RUN=1)
	bash scripts/sync_bucket.sh push $(SYNC_FLAGS) $(FOLDERS)

data-verify:  ## Check the bucket after the data upload (PACKAGE=path/to/sportlight-data.zip for exact sizes)
	$(PY) scripts/verify_bucket.py $(if $(PACKAGE),--package $(PACKAGE))
