import math

time_ranges = []

start_time = 8.0

while start_time < 17.0:
    if start_time == 16.5:
        start_hour = 16
        start_min = 30
        end_hour = 17
        end_min = 0
    else:
        start_hour = math.floor(start_time)
        start_min = (start_time % 1) * 60
        end_hour = start_hour # because of the 10-minute gap, will never spill over except at 5:00
        end_min = start_min + 20
    
    if start_hour < 12:
        start_ampm = "AM"
    if start_hour == 12:
        start_ampm = "PM"
    elif start_hour > 12:
        start_hour = start_hour - 12
        start_ampm = "PM"
    if end_hour < 12:
        end_ampm = "AM"
    if end_hour == 12:
        end_ampm = "PM"
    elif end_hour > 12:
        end_hour = end_hour - 12
        end_ampm = "PM"
    
    start_hour = str(int(start_hour))
    start_min = str(int(start_min))
    end_hour = str(int(end_hour))
    end_min = str(int(end_min))
    
    if start_min == "0":
        start_min = "00"
    if end_min == "0":
        end_min = "00"
    
    time_ranges.append("%s:%s %s to %s:%s %s" % (start_hour, start_min, start_ampm, end_hour, end_min, end_ampm))
    start_time += 0.5

import string

number_of_employees = input("How many employees are there? ")
while int(number_of_employees) >= 26:
    print("Why are you trying to schedule mutually exclusive shifts for so many people?")
    number_of_employees = input("Try again. ")

all_availabilities = {}

for i in range(int(number_of_employees)): # put all the schedule entry stuff in here, then evaluate later
    name = input("Employee's name: ")
    avail = []
    for j in range(len(time_ranges)):
        answer = ""
        while answer not in ["y", "n"]:
            answer = input("Is this employee available from %s? " % time_ranges[j])
        avail.append(answer)
    all_availabilities[name] = avail



# shifts must be at least 50 mins
# no two employees on same shift
