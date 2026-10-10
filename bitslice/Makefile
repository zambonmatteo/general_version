# Model/input paths and other overrides are relative to general_version/.
.PHONY: all host board run flash
all: run
host board run flash:
	$(MAKE) -C .. $@ BACKEND=bitslice
