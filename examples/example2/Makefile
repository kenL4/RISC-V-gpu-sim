ASM = $(wildcard *.s)
OBJ = $(ASM:.s=.o)

CC = riscv64-unknown-elf-gcc
CFLAGS = -Wall -march=rv64i -mabi=lp64

.PHONY: bin asm clean help

# Run to generate binary
all: kernel

# Make binary (for now not supporting stdlib)
kernel: $(OBJ)
	$(CC) -nostdlib -o $@ $^

# Assemble ASM into object files
%.o: %.s
	$(CC) $(CFLAGS) -c $< -o $@

# Clean builds
clean:
	rm -f $(OBJ) kernel

# Help
help:
	@echo "Usage: "
	@echo "    all - Compiles/assembles all kernels to RISC-V binary"
	@echo "    clean - Remove all object and ASM files"