PYTHON ?= python
ARM_CC ?= arm-none-eabi-gcc
ARM_OBJCOPY ?= arm-none-eabi-objcopy
ARM_SIZE ?= arm-none-eabi-size
# Relative to the project root; override with make ... SDK_ROOT="C:/path/to/SDK".
SDK_ROOT ?= ../../AmbiqSuite_R4.5.0
JLINK ?= JLink
DEVICE ?= AMAP42KL-KBR
BACKEND ?= scalar
DATA_DIR ?= data
REPEATS ?= 10

ifneq ($(BACKEND),scalar)
ifneq ($(BACKEND),bitslice)
$(error BACKEND must be scalar or bitslice)
endif
endif

OUT = build/$(BACKEND)
BSP = $(SDK_ROOT)/boards/apollo4l_blue_evb/bsp
FLAGS = -mthumb -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard
FLAGS += -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter -ffp-contract=off -ffunction-sections -fdata-sections
FLAGS += -DPART_apollo4l -DAM_PACKAGE_BGA -DAM_PART_APOLLO4L -Dgcc -DBENCH_REPEATS=$(REPEATS)
INCLUDES = -Ibase_structure -I$(OUT) -I"$(SDK_ROOT)" -I"$(BSP)"
INCLUDES += -I"$(SDK_ROOT)/CMSIS/ARM/Include" -I"$(SDK_ROOT)/CMSIS/AmbiqMicro/Include"
INCLUDES += -I"$(SDK_ROOT)/devices" -I"$(SDK_ROOT)/mcu/apollo4l" -I"$(SDK_ROOT)/mcu/apollo4l/hal" -I"$(SDK_ROOT)/utils"
SOURCES = board/main.c board/startup_gcc.c $(OUT)/data.S
SOURCES += $(addprefix base_structure/,lut.c layer.c encoding.c model.c)
SOURCES += "$(SDK_ROOT)/utils/am_util_delay.c" "$(SDK_ROOT)/utils/am_util_stdio.c" "$(SDK_ROOT)/utils/am_util_string.c"
LIBRARIES = "$(SDK_ROOT)/mcu/apollo4l/hal/mcu/gcc/bin/libam_hal.a" "$(BSP)/gcc/bin/libam_bsp.a"

ifeq ($(BACKEND),bitslice)
FLAGS += -DUSE_BITSLICE
SOURCES += base_structure/bitslice.c
endif

.PHONY: all prepare board flash clean
all: prepare
	"$(ARM_CC)" $(FLAGS) $(INCLUDES) $(SOURCES) -nostartfiles -static -Wl,--gc-sections,--entry,Reset_Handler -Wl,-T,board/linker_script.ld -Wl,--start-group $(LIBRARIES) -lm -lc -lgcc -lnosys -Wl,--end-group -o $(OUT)/firmware.elf
	"$(ARM_OBJCOPY)" -O binary $(OUT)/firmware.elf $(OUT)/firmware.bin
	"$(ARM_SIZE)" $(OUT)/firmware.elf

prepare:
	"$(PYTHON)" -B tools/generate.py --data-dir "$(DATA_DIR)" --output-dir "$(OUT)"

board: all

flash: all
	cd "$(OUT)" && "$(JLINK)" -Device "$(DEVICE)" -If SWD -Speed 4000 -AutoConnect 1 -ExitOnError 1 -NoGui 1 -CommandFile ../../board/flash.jlink

clean:
	"$(PYTHON)" -c "from pathlib import Path; import shutil, sys; root=Path.cwd().resolve(); target=(root/'$(OUT)').resolve(); target.parent == (root/'build').resolve() or sys.exit('Refusing to clean outside build'); shutil.rmtree(target, ignore_errors=True)"
