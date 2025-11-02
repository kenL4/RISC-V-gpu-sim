    .text
    .global _start

_start:
    li t0, 0
    bne t0, zero, end
    addi t0, t0, 1

end:
    addi t0, t0, 1
    ecall