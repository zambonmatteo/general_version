# Model/input paths and other overrides are relative to general_version/.
.PHONY: all host board
all: host
host board:
	$(MAKE) -C .. $@ BACKEND=bitslice
