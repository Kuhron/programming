cell = "B17"  # paste the formula in J3 and then copy to the rest of column J, transforming column I from number into grade letter

conditions = []
equality_conditions = []
between_conditions = []

for grade_row in range(2, 15):
    if_condition = f"{cell}=$B${grade_row}"
    then = f"$A${grade_row}"
    equality_conditions.append((if_condition, then))

for grade_row_lower in range(3, 15):
    grade_row_upper = grade_row_lower - 1
    between_condition = f"AND($B${grade_row_lower}<{cell},{cell}<$B${grade_row_upper})"
    then = f"IF(ABS($B${grade_row_lower}-{cell})<ABS($B${grade_row_upper}-{cell}),$A${grade_row_lower},$A${grade_row_upper})"
    between_conditions.append((between_condition, then))

conditions = equality_conditions + between_conditions

s = f"IF({cell}<>\"\",ENDPOINT,\"---\")"
for if_cond, then in conditions:
    s = s.replace("ENDPOINT", f"IF({if_cond},{then},ENDPOINT)")

s = "=" + s.replace("ENDPOINT", "\"ERROR!\"")

print(s)
