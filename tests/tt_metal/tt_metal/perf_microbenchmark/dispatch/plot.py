import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Raw benchmark data as a string
data = """
BM_pgm_dispatch/n_common_args__kernel_cycles/0/0/manual_time 620526912
BM_pgm_dispatch/n_common_args__kernel_cycles/0/256/manual_time 630392413
BM_pgm_dispatch/n_common_args__kernel_cycles/0/512/manual_time 610601152
BM_pgm_dispatch/n_common_args__kernel_cycles/0/768/manual_time 606716988
BM_pgm_dispatch/n_common_args__kernel_cycles/0/1024/manual_time 607927883
BM_pgm_dispatch/n_common_args__kernel_cycles/64/0/manual_time 621551818
BM_pgm_dispatch/n_common_args__kernel_cycles/64/256/manual_time 667211924
BM_pgm_dispatch/n_common_args__kernel_cycles/64/512/manual_time 633852167
BM_pgm_dispatch/n_common_args__kernel_cycles/64/768/manual_time 649200506
BM_pgm_dispatch/n_common_args__kernel_cycles/64/1024/manual_time 655886018
BM_pgm_dispatch/n_common_args__kernel_cycles/128/0/manual_time 620085762
BM_pgm_dispatch/n_common_args__kernel_cycles/128/256/manual_time 636310488
BM_pgm_dispatch/n_common_args__kernel_cycles/128/512/manual_time 630946019
BM_pgm_dispatch/n_common_args__kernel_cycles/128/768/manual_time 612016815
BM_pgm_dispatch/n_common_args__kernel_cycles/128/1024/manual_time 635356980
BM_pgm_dispatch/n_common_args__kernel_cycles/192/0/manual_time 623423858
BM_pgm_dispatch/n_common_args__kernel_cycles/192/256/manual_time 621896185
BM_pgm_dispatch/n_common_args__kernel_cycles/192/512/manual_time 631060668
BM_pgm_dispatch/n_common_args__kernel_cycles/192/768/manual_time 614590544
BM_pgm_dispatch/n_common_args__kernel_cycles/192/1024/manual_time 623832148
BM_pgm_dispatch/n_common_args__kernel_cycles/256/0/manual_time 616828495
BM_pgm_dispatch/n_common_args__kernel_cycles/256/256/manual_time 620584310
BM_pgm_dispatch/n_common_args__kernel_cycles/256/512/manual_time 609314696
BM_pgm_dispatch/n_common_args__kernel_cycles/256/768/manual_time 616053918
BM_pgm_dispatch/n_common_args__kernel_cycles/256/1024/manual_time 611798524
"""

# Parse the benchmark data
records = []
for line in data.strip().splitlines():
    match = re.match(r".*?(\d+)/(\d+)/manual_time\s+(\d+)", line)
    if match:
        n_common_args = int(match.group(1))
        kernel_cycles = int(match.group(2))
        time_ns = int(match.group(3))
        records.append((n_common_args, kernel_cycles, time_ns))

# Create DataFrame
df = pd.DataFrame(records, columns=["n_common_args", "kernel_cycles", "time_ns"])

# Export data to CSV
df.to_csv("benchmark_results.csv", index=False)

# Plotting
plt.figure(figsize=(10, 6))
palette = sns.color_palette("tab10", n_colors=len(df["n_common_args"].unique()))
sns.lineplot(data=df, x="kernel_cycles", y="time_ns", hue="n_common_args", palette=palette, marker="o")

plt.title("Benchmark Time vs Kernel Cycles for Different n_common_args")
plt.xlabel("Kernel Cycles")
plt.ylabel("Time (ns)")
plt.grid(True)
plt.tight_layout()

# Save plot to PNG file
plt.savefig("benchmark_plot.png", dpi=300)
plt.show()
