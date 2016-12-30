from datetime import datetime, timedelta, date

# the strangely-named West Memphis Radio Rolling clock convention from a dream I had in high school
# minutes:seconds, where minutes have 90 seconds and a day has 960 minutes

def wmrr_from_datetime(dt, offset):
	try:
		dt = dt.replace(microsecond=0).time()
	except AttributeError:
		dt = dt.replace(microsecond=0)
	dt = (datetime.combine(date.today(), dt) + timedelta(hours=offset)).time()
	midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
	td = datetime.combine(date.today(), dt) - datetime.combine(date.today(), midnight)
	s = td.total_seconds()
	a, b = s // 90, s % 90
	return "{0}:{1}".format(str(int(a)).rjust(3, "0"), str(int(b)).rjust(2, "0"))


offset = int(input("UTC offset (hours): "))

t0 = None
while True:
	t = datetime.now().replace(microsecond=0).time()
	if t0 is None or t != t0:
		wmrr = wmrr_from_datetime(t, offset)
		print(wmrr, end="\r")
		t0 = t