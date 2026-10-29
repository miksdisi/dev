"""
Schedular v2.0 — Оптимизатор (HiGHS через scipy)

Этап 1: время  — какие уроки в какие слоты.
        Решается по частям (LNS — поиск по большим окрестностям):
        1) строим расписание группами по GROUP_SIZE классов;
        2) многократно «отпускаем» группу связанных классов и переоптимизируем её
           точно, пока остальное расписание зафиксировано.
Этап 2: кабинеты — урок за уроком, задача о назначениях (венгерский алгоритм).
"""
import random
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from milp_model import Model
from data import (DAYS, SLOTS_PER_DAY, CLASSES, ASSIGNMENTS, PREFERENCES, ROOMS,
                  CLASS_SIZE, HOME_ROOM, count_ok_slots, total_load)

# ── Настройки ──
PREF_WEIGHTS = {1: 10, -1: -10, -10: -1000}   # могу / нежелательно / не ставить (нейтрально = 0)
W_BLOCK = 1                    # штраф за каждый отдельный «кусок» занятости учителя в день (меньше кусков = меньше окон)
W_STAY = 10                    # бонус, если учитель остаётся в том же кабинете на следующем уроке
MAX_SAME_SUBJECT_PER_DAY = 2   # не больше N уроков одного предмета у класса в день
GROUP_SIZE = 8                 # сколько классов переоптимизируется за раз
TIME_LIMIT_TIME = 120          # секунд на улучшение расписания по времени (этап 1)
TIME_LIMIT_SUB = 10            # секунд на одну подзадачу этапа 1
SEED = 0

SLOT_NUMS = range(1, SLOTS_PER_DAY + 1)

# Назначения: units[k] = (учитель, предмет, класс, часов)
UNITS = [(t, subj, cls, h) for t, a in ASSIGNMENTS.items() for subj, cls, h in a]
CLASS_TEACHERS = {}    # класс -> учителя
TEACHER_CLASSES = {}   # учитель -> классы
for t, _, cls, _ in UNITS:
    CLASS_TEACHERS.setdefault(cls, set()).add(t)
    TEACHER_CLASSES.setdefault(t, set()).add(cls)


# ── Вспомогательные функции ──

def follows_150_rule(teacher):
    """Анкета учитывается, только если 'могу' >= 150% нагрузки."""
    return count_ok_slots(teacher) >= 1.5 * total_load(teacher)


def pref_weight(teacher, d, s):
    """Вклад одного урока учителя в слоте (d, s) в целевую функцию."""
    if not follows_150_rule(teacher):
        return 0
    return PREF_WEIGHTS.get(PREFERENCES[teacher].get((d, s), 0), 0)


def room_score(teacher, cls, room):
    """Свой кабинет = 3, кабинет класса = 2, общий = 1, чужой = 0."""
    owner = ROOMS[room][1]
    if owner == teacher:
        return 3
    if room == HOME_ROOM.get(cls):
        return 2
    return 1 if owner == "общий" else 0


def transitions(lessons):
    """Число смен кабинета за день. lessons = [(урок, кабинет), ...]"""
    rooms = [r for _, r in sorted(lessons)]
    return sum(a != b for a, b in zip(rooms, rooms[1:]))


def count_transitions(schedule):
    days = {}
    for t, _, _, d, s, r in schedule:
        days.setdefault((t, d), []).append((s, r))
    return sum(transitions(l) for l in days.values())


def time_objective(lessons):
    """Качество расписания по времени (больше = лучше). lessons = [(t, subj, cls, d, s), ...]"""
    busy = {(t, d, s) for t, _, _, d, s in lessons}
    blocks = sum(1 for (t, d, s) in busy if (t, d, s - 1) not in busy)
    return sum(pref_weight(t, d, s) for t, _, _, d, s in lessons) - W_BLOCK * blocks


# ── Главная функция ──

def solve():
    """Возвращает (schedule, score).
    schedule = [(teacher, subject, class, day, slot, room), ...]
    score = (баллы по анкетам, -переходы, баллы по кабинетам)"""
    lessons = solve_time()
    schedule = solve_rooms(lessons)
    score = (sum(PREFERENCES[t].get((d, s), 0) for t, _, _, d, s, _ in schedule),
             -count_transitions(schedule),
             sum(room_score(t, c, r) for t, _, c, _, _, r in schedule))
    return schedule, score


# ── Этап 1: время ──

def solve_time():
    rng = random.Random(SEED)

    # 1) Построение: группами классов, каждая следующая группа видит уже занятых учителей
    lessons = []
    for i in range(0, len(CLASSES), GROUP_SIZE):
        group = CLASSES[i:i + GROUP_SIZE]
        new = _solve_classes(group, lessons)
        if new is None:
            raise RuntimeError(f"Этап 1: не удалось построить расписание для классов {group}.")
        lessons += new

    # 2) Улучшение: переоптимизируем группы связанных классов, пока есть время
    if len(CLASSES) > GROUP_SIZE:
        deadline = time.time() + TIME_LIMIT_TIME
        best = time_objective(lessons)
        while time.time() < deadline:
            group = _related_classes(rng)
            rest = [l for l in lessons if l[2] not in group]
            new = _solve_classes(group, rest)
            if new is not None and time_objective(rest + new) >= best:
                lessons, best = rest + new, time_objective(rest + new)

    return lessons


def _related_classes(rng):
    """Случайная группа классов, связанных общими учителями."""
    group = []
    t = rng.choice(list(TEACHER_CLASSES))
    for _ in range(50):
        new = [c for c in TEACHER_CLASSES[t] if c not in group]
        rng.shuffle(new)
        group += new[:GROUP_SIZE - len(group)]
        if len(group) >= GROUP_SIZE:
            break
        t = rng.choice([u for c in group for u in CLASS_TEACHERS[c]])
    return group


def _solve_classes(classes, fixed):
    """Точно расставляет уроки классов `classes`, не трогая уроки `fixed`.
    Возвращает список уроков [(t, subj, cls, d, s), ...] или None."""
    m = Model()
    busy = {(t, d, s) for t, _, _, d, s in fixed}       # учитель уже занят другим классом
    ks = [k for k, u in enumerate(UNITS) if u[2] in classes]

    # x[k, d, s] = 1, если урок назначения k стоит в день d на урок s
    x = {}
    for k in ks:
        t, _, _, h = UNITS[k]
        for d in DAYS:
            for s in SLOT_NUMS:
                if (t, d, s) not in busy:
                    x[k, d, s] = m.var(cost=-pref_weight(t, d, s))
        m.add([(x[k, d, s], 1) for d in DAYS for s in SLOT_NUMS if (k, d, s) in x], h, h)
        for d in DAYS:
            m.add([(x[k, d, s], 1) for s in SLOT_NUMS if (k, d, s) in x], hi=MAX_SAME_SUBJECT_PER_DAY)

    def at(group, d, s, coef=1):
        """Сумма x по назначениям group в слоте (d, s)."""
        return [(x[k, d, s], coef) for k in group if (k, d, s) in x]

    # Классы: один урок в слот, без окон, начинают с 1-го урока, равномерно по дням
    for cls in classes:
        cks = [k for k in ks if UNITS[k][2] == cls]
        H = sum(UNITS[k][3] for k in cks)
        lo, hi = H // len(DAYS), -(-H // len(DAYS)) + 1
        for d in DAYS:
            for s in SLOT_NUMS:
                m.add(at(cks, d, s), hi=1)
                if s > 1:  # занят на уроке s  =>  занят на уроке s-1
                    m.add(at(cks, d, s) + at(cks, d, s - 1, -1), hi=0)
            m.add([v for s in SLOT_NUMS for v in at(cks, d, s)], lo, hi)

    # Учителя: один урок в слот, штраф за каждый новый «кусок» занятости в день
    for t in {UNITS[k][0] for k in ks}:
        tks = [k for k in ks if UNITS[k][0] == t]
        for d in DAYS:
            for s in SLOT_NUMS:
                m.add(at(tks, d, s), hi=1)
                # start >= занят(s) - занят(s-1); занятость = уроки этой группы + уже зафиксированные
                was = (t, d, s - 1) in busy
                now = (t, d, s) in busy
                start = m.var(cost=W_BLOCK)
                prev = at(tks, d, s - 1, -1) if s > 1 else []
                m.add(at(tks, d, s) + prev + [(start, -1)], hi=int(was) - int(now))

    sol = m.solve(TIME_LIMIT_SUB)
    if sol is None:
        return None
    return [(UNITS[k][0], UNITS[k][1], UNITS[k][2], d, s) for (k, d, s), v in x.items() if sol[v]]


# ── Этап 2: кабинеты ──

def solve_rooms(lessons):
    """Кабинеты распределяются урок за уроком (слот за слотом).
    В каждом слоте — задача о назначениях «уроки -> кабинеты», которую венгерский
    алгоритм решает точно. Выгода варианта: тип кабинета + бонус, если учитель
    остаётся там же, где был на своём предыдущем уроке сегодня."""
    rooms = list(ROOMS)
    schedule = []
    for d in DAYS:
        last_room = {}   # учитель -> кабинет его предыдущего урока сегодня
        for s in SLOT_NUMS:
            now = [l for l in lessons if l[3] == d and l[4] == s]
            if not now:
                continue
            gain = np.full((len(now), len(rooms)), -1e6)      # -1e6 = класс не помещается
            for i, (t, _, cls, _, _) in enumerate(now):
                for j, r in enumerate(rooms):
                    if ROOMS[r][0] >= CLASS_SIZE[cls]:
                        gain[i, j] = room_score(t, cls, r) + W_STAY * (last_room.get(t) == r)
            rows, cols = linear_sum_assignment(gain, maximize=True)
            if (gain[rows, cols] < -1e5).any():
                raise RuntimeError(f"Этап 2: не хватает подходящих кабинетов ({d}, урок {s}).")
            for i, j in zip(rows, cols):
                last_room[now[i][0]] = rooms[j]
                schedule.append((*now[i], rooms[j]))
    return schedule
