PYTHON ?= python
CC = gcc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Werror -ffp-contract=off
MODEL_DIR ?= model
ARM_CC ?= arm-none-eabi-gcc
SDK_ROOT ?= $(if $(AMBIQ_SDK_ROOT),$(AMBIQ_SDK_ROOT),../../AmbiqSuite_R4.5.0)
REPEATS ?= 10
BACKEND ?= scalar
JLINK ?= JLink
DEVICE ?= AMAP42KL-KBR
JLINK_SERIAL ?=

.PHONY: all host board run flash prepare encode clean
all: run

host:
	"$(PYTHON)" -B tools/run.py --target host --backend "$(BACKEND)" --model-dir "$(MODEL_DIR)" --cc "$(CC)" --cflags "$(CFLAGS)" --repeats "$(REPEATS)"

prepare:
	"$(PYTHON)" -B tools/run.py --target prepare --backend "$(BACKEND)" --model-dir "$(MODEL_DIR)" --cc "$(CC)" --cflags "$(CFLAGS)" --repeats "$(REPEATS)"

board:
	"$(PYTHON)" -B tools/run.py --target board --backend "$(BACKEND)" --model-dir "$(MODEL_DIR)" --cc "$(CC)" --cflags "$(CFLAGS)" --arm-cc "$(ARM_CC)" --sdk-root "$(SDK_ROOT)" --repeats "$(REPEATS)"

run:
	"$(PYTHON)" -B tools/run.py --target run --backend "$(BACKEND)" --model-dir "$(MODEL_DIR)" --cc "$(CC)" --cflags "$(CFLAGS)" --arm-cc "$(ARM_CC)" --sdk-root "$(SDK_ROOT)" --repeats "$(REPEATS)" --jlink "$(JLINK)" --device "$(DEVICE)" --jlink-serial "$(JLINK_SERIAL)"

flash:
	"$(PYTHON)" -B tools/run.py --target flash --jlink "$(JLINK)" --device "$(DEVICE)" --jlink-serial "$(JLINK_SERIAL)"

encode:
	"$(PYTHON)" -B tools/run.py --target encode --model-dir "$(MODEL_DIR)" --cc "$(CC)" --cflags "$(CFLAGS)"

clean:
	"$(PYTHON)" -B tools/run.py --target clean
