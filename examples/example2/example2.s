    .text
    .global _start

_start:
    jal x0, 12
    jal x1, 8
    jal x2, 4
    lw x0, 10(x1)
    addi x0, x10, 10