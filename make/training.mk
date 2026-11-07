# Training agent targets. Placeholders until the training agent lands T0 to T4.
.PHONY: build-% train-% eval-%

define TODO
	@echo "$@: not implemented yet (owner: $(1) agent, milestone $(2))" && exit 1
endef

build-%:  ## Source + featurize + label one sport (training agent, T1-T4)
	$(call TODO,training,T1-T4)

train-%:  ## Train and calibrate one sport (training agent, T1-T3)
	$(call TODO,training,T1-T3)

eval-%:  ## Evaluate one sport on test (training agent, T1-T3)
	$(call TODO,training,T1-T3)
