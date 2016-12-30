import time

def get_tempo_bpm(timestamps):
	if len(timestamps) <= 1:
		return float("nan")
	# diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
	# return sum(diffs) * 1.0/len(diffs) * 60
	t_range = max(timestamps) - min(timestamps)
	return (len(timestamps) - 1) * 1.0/t_range * 60

n = 10
print("Press enter at the tempo desired. The last {n} beats will be used to calculate tempo.".format(n=n))
ts = []
while True:
	input()
	t = time.time()
	ts.append(t)
	ts_reduced = ts if len(ts) <= 11 else ts[len(ts)-10:]
	tempo = get_tempo_bpm(ts_reduced)
	print("{tempo:.2f} bpm".format(tempo=tempo))
