# coding=utf-8
import math
from pydantic import BaseModel


def are_close(a, b, tolerance=0.0):
    return math.isclose(a, b, abs_tol=tolerance)


class Point(BaseModel):
    x: float
    y: float

    @classmethod
    def from_tuple(cls, vals: tuple[float, float]):
        x, y = vals
        return cls(x=x, y=y)

    def to_tuple(self) -> tuple[float, float]:
        return self.x, self.y


class Interval(BaseModel):
    start: float | None
    end: float | None

    @property
    def width(self):
        if self.is_empty():
            return 0
        return self.end - self.start

    def __contains__(self, item: float):
        return self.start <= item <= self.end

    def is_empty(self):
        return self.start is None and self.end is None

    @staticmethod
    def empty_value():
        return Interval(start=None, end=None)

    @staticmethod
    def from_tuple(t: tuple[float, float]):
        start, end = t
        return Interval(start=start, end=end)

    def to_tuple(self) -> tuple[float, float]:
        return self.start, self.end

    @staticmethod
    def intersection(a: "Interval", b: "Interval"):
        if a.start > b.end or b.start > a.end:
            return Interval.empty_value()
        start = max(a.start, b.start)
        end = min(a.end, b.end)
        return Interval(start=start, end=end)

    @staticmethod
    def intersection_of_values(a: tuple[float, float], b: tuple[float, float]):
        a = Interval.from_tuple(a)
        b = Interval.from_tuple(b)
        return Interval.intersection(a, b)

    @staticmethod
    def covers(a: "Interval", b: "Interval"):
        return a.start <= b.start and b.end <= a.end


class Rectangle(BaseModel):
    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def from_tuple(cls, values: tuple[float, float, float, float]):
        """
        elements order: (left, top, right, bottom)
        :param values:
        :return:
        """
        left, top, right, bottom = values
        if left > right:
            raise ValueError('`left` should be less than or equal to `right`')
        if top > bottom:
            raise ValueError('`top` should be less than or equal to `bottom`')
        return cls(left=left,
                   top=top,
                   right=right,
                   bottom=bottom)

    def to_tuple(self) -> tuple[float, float, float, float]:
        return self.left, self.top, self.right, self.bottom

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        return self.width * self.height

    def intersection(self, r2: "Rectangle"):
        left = max(self.left, r2.left)
        top = max(self.top, r2.top)

        right = min(self.right, r2.right)
        bottom = min(self.bottom, r2.bottom)

        if left > right or top > bottom:
            return None
        return self.from_tuple((left, top, right, bottom))



def factorial(n: int):
    print(n)
    if n <= 1:
        return 1
    return n * factorial(n-1)


def ex(x, n):
    def part(np):
        return x ** np / factorial(np)

    return 1 + sum(part(i) for i in range(1, n+1))


if __name__ == '__main__':
    print(ex(0.1, 5), math.exp(0.1))
