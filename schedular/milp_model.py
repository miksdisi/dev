"""
Мини-обёртка над scipy.optimize.milp (решатель HiGHS).
Все переменные — 0/1, целевая функция минимизируется.
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix


class Model:
    def __init__(self):
        self.cost = []                                   # цена каждой переменной
        self.rows, self.cols, self.vals = [], [], []     # разреженная матрица ограничений
        self.lo, self.hi = [], []                        # границы ограничений

    def var(self, cost=0.0):
        """Новая переменная 0/1. cost — её вклад в целевую функцию (минимизируем)."""
        self.cost.append(cost)
        return len(self.cost) - 1

    def add(self, terms, lo=-np.inf, hi=np.inf):
        """Ограничение: lo <= sum(coef * var) <= hi.  terms = [(var, coef), ...]"""
        r = len(self.lo)
        for v, a in terms:
            self.rows.append(r)
            self.cols.append(v)
            self.vals.append(a)
        self.lo.append(lo)
        self.hi.append(hi)

    def solve(self, time_limit=60):
        """Возвращает массив значений 0/1 или None, если решение не найдено."""
        n = len(self.cost)
        A = csr_matrix((self.vals, (self.rows, self.cols)), shape=(len(self.lo), n))
        res = milp(
            c=np.array(self.cost, dtype=float),
            constraints=LinearConstraint(A, self.lo, self.hi),
            integrality=np.ones(n),
            bounds=Bounds(0, 1),
            options={"time_limit": time_limit},
        )
        self.result = res   # статус, зазор до оптимума и т.п. — для диагностики
        if res.x is None:
            return None
        return np.round(res.x).astype(int)
