def find_average(number: int) -> int:
    if (number % 2) == 0:
        return int(number / 2)
    else:
        return int((number - 1) / 2)


def divide_on_4(number: int):
    b: int = find_average(number)
    a: int = find_average(b)
    c: int = b + 1
    d: int = find_average(number + b)
    return int(a), int(b), int(c), int(d)


a, b, c, d = divide_on_4(8)
assert a == 2
assert b == 4
assert c == 5
assert d == 6

a, b, c, d = divide_on_4(11)
assert a == 2
assert b == 5
assert c == 6
assert d == 8
