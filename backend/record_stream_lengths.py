import subprocess
import pandas as pd
import time
from datetime import datetime

start_time = time.time()
duration = 10 * 60 # 10 minutes
interval = 2 # 2 seconds

csv_filename = "stream_lengths.csv"
all_data = []

print(f"Starting data collection for {duration/60} minutes every {interval} seconds...")
print(f"Output will be saved to {csv_filename}")

while time.time() - start_time < duration:
    loop_start = time.time()
    
    # Run the user's bash command
    cmd = """docker exec redis redis-cli KEYS "stream:cam:*" | tr -d '\r' | while read key; do [ ! -z "$key" ] && echo "$key: $(docker exec redis redis-cli XLEN "$key")"; done"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        
        row = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        for line in lines:
            if line and ':' in line:
                key, val = line.rsplit(':', 1)
                row[key.strip()] = int(val.strip())
                
        if len(row) > 1: # Make sure we got some data
            all_data.append(row)
            
            # Save to DataFrame and CSV
            df = pd.DataFrame(all_data)
            df.to_csv(csv_filename, index=False)
            print(f"[{row['timestamp']}] Collected data. Total rows: {len(df)}")
        else:
            print(f"[{row['timestamp']}] No data collected.")
            
    except Exception as e:
        print(f"Error: {e}")
        
    elapsed = time.time() - loop_start
    sleep_time = max(0, interval - elapsed)
    time.sleep(sleep_time)

print("Finished data collection. Final DataFrame saved to", csv_filename)
df = pd.DataFrame(all_data)
print(df.head())
