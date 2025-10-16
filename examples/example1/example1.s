    .text
    .globl _start

_start:
    addi t0, t0, 1
    lw t1, 1(t1)
    addi t1, t1, 1
    beq zero, t0, 0 # Loop if 0 == 1
    srli t1, t1, 1
    sub t2, t1, t0
    xor t2, t0, t2
    xor t0, t0, t2
    xor t2, t0, t2
