python3 -c "import time; f=lambda n: n if n <= 0 else f(n-1) if time.sleep(0.1) else f(n-1) if not print(hash(time.time())) else None; f(950);" &
pid_to_kill=$!
sleep 5
kill $pid_to_kill
echo "Sacrificed"

