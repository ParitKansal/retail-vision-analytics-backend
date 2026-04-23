import subprocess
import time
import re
from collections import defaultdict

DURATION_SECONDS = 30
SAMPLE_INTERVAL = 1

stats_data = defaultdict(list)

print(f"🔄 Monitoring CPU usage for {DURATION_SECONDS} seconds... Please wait.")

start_time = time.time()
iteration = 0

try:
    while time.time() - start_time < DURATION_SECONDS:
        iteration += 1
        # Get a snapshot
        result = subprocess.run(
            ['docker', 'stats', '--no-stream', '--format', '{{.Name}},{{.CPUPerc}}'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error reading stats: {result.stderr}")
            break

        lines = result.stdout.strip().split('\n')
        for line in lines:
            if not line: continue
            try:
                name, cpu_str = line.split(',')
                # Remove % and convert to float
                cpu_val = float(cpu_str.replace('%', '').strip())
                stats_data[name].append(cpu_val)
            except ValueError:
                continue
        
        # Simple progress indicator
        print(f".", end="", flush=True)
        time.sleep(SAMPLE_INTERVAL)

    print("\n\n📊 RESULTS (Avg / Max over 30s):")
    print(f"{'SERVICE NAME':<45} | {'AVG CPU %':<10} | {'MAX CPU %':<10}")
    print("-" * 75)

    # Sort by MAX CPU usage descending
    sorted_stats = sorted(stats_data.items(), key=lambda item: max(item[1]) if item[1] else 0, reverse=True)

    for name, values in sorted_stats:
        if not values: continue
        avg_cpu = sum(values) / len(values)
        max_cpu = max(values)
        print(f"{name:<45} | {avg_cpu:8.2f}% | {max_cpu:8.2f}%")

except KeyboardInterrupt:
    print("\nStopped.")
