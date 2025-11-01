    .text
    .global _start

_start:
    li t0, 123
    sw t0, 0(t1)
    lw t2, 0(t1)