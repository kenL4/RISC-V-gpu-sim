    .text
    .globl _start

_start:
    addi t0, t0, 1
    addi t0, t0, 1
    sub t2, t1, t0
    xor t2, t0, t2
    xor t0, t0, t2
    xor t2, t0, t2
    neg t0, t0
    srli t0, t0, 1
    sll t0, t0, t2