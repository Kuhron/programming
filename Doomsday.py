import datetime
import random
import time


DAYS = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]


def get_day_using_doomsday_algorithm(date):
    year = date.year
    month = date.month
    day = date.day
    century, y = divmod(year, 100)
    anchor = [2, 0, 5, 3][century % 4]
    a, b = divmod(y, 12)
    c = b // 4
    doomsday = (anchor + a + b + c) % 7

    regular_doomsdays = {
        3: 0,
        4: 4,
        5: 9,
        6: 6,
        7: 11,
        8: 8,
        9: 5,
        10: 10,
        11: 7,
        12: 12
    }

    if month == 1:
        d = 4 if is_leap_year(year) else 3
    elif month == 2:
        d = 29 if is_leap_year(year) else 28
    else:
        d = regular_doomsdays[month]

    diff = day - d
    return (doomsday + diff) % 7


def is_leap_year(year):
    if year % 4 != 0:
        return False
    if year % 100 == 0:
        return (year % 100) % 4 == 0
    return True


def get_day_using_datetime(date):
    return (date.weekday() + 1) % 7


def day_to_number(s):
    return DAYS.index(s)


def number_to_day(n):
    return DAYS[n]


def get_random_date():
    year = random.randint(1800, 2199)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    date = datetime.date(year, month, day)
    return date


def test():
    for _ in range(10):
        date = get_random_date()
        assert get_day_using_datetime(date) == get_day_using_doomsday_algorithm(date)


def practice():
    t0 = time.time()
    score = 0
    total = 0
    for _ in range(5):
        date = get_random_date()
        answer = input("What day of the week (index number) was {}?\n".format(date.strftime("%Y-%m-%d")))
        acceptable = False
        while not acceptable:
            try:
                int(answer)
                acceptable = True
            except ValueError:
                print("invalid answer! (use digits 0-6)")
                answer = input("try again: ")
        day = get_day_using_datetime(date)
        assert get_day_using_doomsday_algorithm(date) == day
        right = int(answer) == day
        if right:
            print("correct!")
            score += 1
        else:
            print("doh! The correct answer was {} ({}).".format(day, number_to_day(day)))
        total += 1
    t = time.time() - t0
    print("final score: {} / {}; total time: {:.2f} seconds ({:.2f} average per right answer)".format(score, total, t, t / score if score > 0 else float("nan")))


if __name__ == "__main__":
    test()
    practice()
