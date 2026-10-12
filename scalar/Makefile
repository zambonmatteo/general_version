# Keep variant entry points separate, while sharing sources and model data.
.PHONY: all host board run flash
all: run
host board run flash:
	$(MAKE) -C .. $@ BACKEND=scalar
