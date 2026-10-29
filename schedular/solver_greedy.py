"""
Schedular v1.3 — Жадный солвер (оставлен для сравнения)
Жадный алгоритм с перезапусками.
Приоритеты при выборе (слот, кабинет): анкета > меньше переходов > тип кабинета.
"""
import random
from data import (DAYS, SLOTS_PER_DAY, ASSIGNMENTS, PREFERENCES, ROOMS,
                  CLASS_SIZE, HOME_ROOM, count_ok_slots)


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
    """Суммарное число переходов всех учителей за неделю."""
    days = {}
    for t, _, _, d, s, r in schedule:
        days.setdefault((t, d), []).append((s, r))
    return sum(transitions(l) for l in days.values())


def solve(num_restarts=50):
    """Возвращает (schedule, score).
    schedule = [(teacher, subject, class, day, slot, room), ...]
    score = (баллы по анкетам, -переходы, баллы по кабинетам) — сравнивается по порядку."""
    best, best_score = None, None
    for i in range(num_restarts):
        sch = _greedy(seed=i)
        if sch is None:
            continue
        sc = (sum(PREFERENCES[t].get((d, s), 0) for t, _, _, d, s, _ in sch),
              -count_transitions(sch),
              sum(room_score(t, c, r) for t, _, c, _, _, r in sch))
        if best_score is None or sc > best_score:
            best, best_score = sch, sc
    return best, best_score


def _greedy(seed=None):
    rng = random.Random(seed)

    # Разворачиваем назначения в отдельные уроки
    tasks = []
    for t, assigns in ASSIGNMENTS.items():
        for subj, cls, cnt in assigns:
            tasks += [(t, subj, cls)] * cnt

    # Самые ограниченные учителя — первыми (+ шум для разнообразия)
    tasks.sort(key=lambda x: count_ok_slots(x[0]) + rng.random() * 2)

    teacher_busy = set()  # (teacher, day, slot)
    class_busy = set()    # (class, day, slot)
    room_busy = set()     # (room, day, slot)
    teacher_day = {}      # (teacher, day) -> [(slot, room), ...]
    schedule = []

    for t, subj, cls in tasks:
        fit_rooms = [r for r, (cap, _) in ROOMS.items() if cap >= CLASS_SIZE[cls]]

        options = [
            (d, s, r)
            for d in DAYS for s in range(1, SLOTS_PER_DAY + 1)
            if (t, d, s) not in teacher_busy and (cls, d, s) not in class_busy
            for r in fit_rooms if (r, d, s) not in room_busy
        ]
        if not options:
            return None

        def key(o):
            d, s, r = o
            day = teacher_day.get((t, d), [])
            extra = transitions(day + [(s, r)]) - transitions(day)  # сколько переходов добавит урок
            return (PREFERENCES[t].get((d, s), 0), -extra, room_score(t, cls, r), rng.random())

        d, s, r = max(options, key=key)

        teacher_busy.add((t, d, s))
        class_busy.add((cls, d, s))
        room_busy.add((r, d, s))
        teacher_day.setdefault((t, d), []).append((s, r))
        schedule.append((t, subj, cls, d, s, r))

    return schedule
