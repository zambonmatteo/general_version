# Keep variant entry points separate, while sharing sources and model data.
.PHONY: all host board
all: host
host board:
	$(MAKE) -C .. $@ BACKEND=scalar
