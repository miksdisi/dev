"""Schedular — точка входа."""
from solver import solve
from render import render

schedule, (pref, neg_moves, rooms) = solve()
print(f"Уроков: {len(schedule)} | анкеты: {pref} | переходов: {-neg_moves} | кабинеты: {rooms}")
render(schedule, "/mnt/user-data/outputs/schedular_result.xlsx")
