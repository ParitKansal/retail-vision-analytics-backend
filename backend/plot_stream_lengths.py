import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('stream_lengths.csv')

cols = [
    'stream:cam:5', 'stream:cam:7', 'stream:cam:6',
    'stream:cam:3', 'stream:cam:4', 'stream:cam:1', 'stream:cam:2'
]

# Create an output directory for the individual plots
output_dir = 'stream_plots'
os.makedirs(output_dir, exist_ok=True)

for col in cols:
    if col in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[col], label=col, color='blue')
        plt.xlabel("Index (Every 2 seconds)")
        plt.ylabel("Stream Length (Pending frames)")
        plt.title(f"{col} - Length Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save each plot to the directory
        safe_filename = col.replace(':', '_') + '.png'
        save_path = os.path.join(output_dir, safe_filename)
        plt.savefig(save_path)
        plt.close() # Close the figure so it doesn't overlap with the next one
        print(f"Saved: {save_path}")

print("All individual plots have been generated in the 'stream_plots/' directory.")
