/******************************************************************************
 *
 * linker_script.ld - Linker script for applications using startup_gnu.c
 *
 *****************************************************************************/
ENTRY(Reset_Handler)

MEMORY
{
    MCU_MRAM     (rx)  : ORIGIN = 0x00018000, LENGTH = 1998848
    MCU_TCM      (rwx) : ORIGIN = 0x10000000, LENGTH = 393216
    SHARED_SRAM  (rwx) : ORIGIN = 0x10060000, LENGTH = 1048576
}

SECTIONS
{
    .text :
    {
        . = ALIGN(4);
        KEEP(*(.isr_vector))
        KEEP(*(.patch))
        *(.text)
        *(.text*)
        *(.rodata)
        *(.rodata*)
        . = ALIGN(4);
        _etext = .;
    } > MCU_MRAM

    .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } > MCU_MRAM
    .ARM.exidx : { *(.ARM.exidx* .gnu.linkonce.armexidx.*) } > MCU_MRAM

    /* User stack section initialized by startup code. */
    .stack (NOLOAD):
    {
        . = ALIGN(8);
        __dwn_stack_start = .;
        *(.stack)
        *(.stack*)
        . = ALIGN(8);
        __dwn_stack_end = .;
    } > MCU_TCM

    .heap (NOLOAD): {
        __heap_start__ = .;
        end = __heap_start__;
        _end = end;
        __end = end;
        KEEP(*(.heap))
        __heap_end__ = .;
        __HeapLimit = __heap_end__;
    } > MCU_TCM

    .data :
    {
        . = ALIGN(4);
        _sdata = .;
        *(.data)
        *(.data*)
        *(.ramfunc) 
        . = ALIGN(4);
        _edata = .;
    } > MCU_TCM AT>MCU_MRAM

    /* used by startup to initialize data */
    _init_data = LOADADDR(.data);

    .bss :
    {
        . = ALIGN(4);
        _sbss = .;
        *(.bss)
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > MCU_TCM

    .shared (NOLOAD):
    {
        . = ALIGN(4);
        KEEP(*(.resource_table))
        KEEP(*(.shared))
        . = ALIGN(4);
    } > SHARED_SRAM AT>MCU_MRAM
    .ARM.attributes 0 : { *(.ARM.attributes) }

    /* Absolute byte counts for the runtime memory report. */
    /* Include alignment gaps in the TCM allocation, excluding the stack.
     * The shared bank is separate; the unused space between banks is not used.
     */
    __dwn_static_ram_bytes = ADDR(.bss) + SIZEOF(.bss) - ORIGIN(MCU_TCM)
                             - SIZEOF(.stack) + SIZEOF(.shared);
    __dwn_heap_bytes = SIZEOF(.heap);
    /* .data is the last initialized section. Include its MRAM copy, unwind
     * tables and image padding; exclude NOLOAD sections and the bootloader.
     * The host independently checks this against every ELF LMA and the BIN.
     */
    __dwn_mram_image_bytes = ABSOLUTE(LOADADDR(.data) + SIZEOF(.data) - ORIGIN(MCU_MRAM));
    ASSERT(SIZEOF(.heap) == 0, "DWN memory accounting requires no heap")
}
