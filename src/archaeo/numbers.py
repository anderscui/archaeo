# coding=utf-8
from math import isqrt


def is_perfect_number(n: int) -> bool:
    if n <= 1:
        return False

    # sum of proper divisors
    aliquot_parts_sum = 1
    for i in range(2, isqrt(n) + 1):
        if n % i == 0:
            aliquot_parts_sum += i
            other = n // i
            if other != i:
                aliquot_parts_sum += other
    return aliquot_parts_sum == n
