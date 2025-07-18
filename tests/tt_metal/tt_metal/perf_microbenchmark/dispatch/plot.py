import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Raw benchmark data as a string
data = """
BM_pgm_dispatch/n_common_args__kernel_cycles/0/0/manual_time  61140915
BM_pgm_dispatch/n_common_args__kernel_cycles/0/64/manual_time 59353570
BM_pgm_dispatch/n_common_args__kernel_cycles/0/128/manual_time  58948392
BM_pgm_dispatch/n_common_args__kernel_cycles/0/192/manual_time  60596660
BM_pgm_dispatch/n_common_args__kernel_cycles/0/256/manual_time  58944371
BM_pgm_dispatch/n_common_args__kernel_cycles/0/320/manual_time  59120287
BM_pgm_dispatch/n_common_args__kernel_cycles/0/384/manual_time  60014516
BM_pgm_dispatch/n_common_args__kernel_cycles/0/448/manual_time  60263471
BM_pgm_dispatch/n_common_args__kernel_cycles/0/512/manual_time  60803252
BM_pgm_dispatch/n_common_args__kernel_cycles/0/576/manual_time  58874772
BM_pgm_dispatch/n_common_args__kernel_cycles/0/640/manual_time  60674948
BM_pgm_dispatch/n_common_args__kernel_cycles/0/704/manual_time  60533468
BM_pgm_dispatch/n_common_args__kernel_cycles/0/768/manual_time  59217539
BM_pgm_dispatch/n_common_args__kernel_cycles/0/832/manual_time  60431453
BM_pgm_dispatch/n_common_args__kernel_cycles/0/896/manual_time  60802074
BM_pgm_dispatch/n_common_args__kernel_cycles/0/960/manual_time  60723414
BM_pgm_dispatch/n_common_args__kernel_cycles/0/1024/manual_time 59163706
BM_pgm_dispatch/n_common_args__kernel_cycles/32/0/manual_time 60279321
BM_pgm_dispatch/n_common_args__kernel_cycles/32/64/manual_time  61020359
BM_pgm_dispatch/n_common_args__kernel_cycles/32/128/manual_time 60790643
BM_pgm_dispatch/n_common_args__kernel_cycles/32/192/manual_time 61185081
BM_pgm_dispatch/n_common_args__kernel_cycles/32/256/manual_time 60602860
BM_pgm_dispatch/n_common_args__kernel_cycles/32/320/manual_time 59129422
BM_pgm_dispatch/n_common_args__kernel_cycles/32/384/manual_time 58699973
BM_pgm_dispatch/n_common_args__kernel_cycles/32/448/manual_time 60562873
BM_pgm_dispatch/n_common_args__kernel_cycles/32/512/manual_time 60514618
BM_pgm_dispatch/n_common_args__kernel_cycles/32/576/manual_time 58953727
BM_pgm_dispatch/n_common_args__kernel_cycles/32/640/manual_time 60576070
BM_pgm_dispatch/n_common_args__kernel_cycles/32/704/manual_time 60843918
BM_pgm_dispatch/n_common_args__kernel_cycles/32/768/manual_time 60740517
BM_pgm_dispatch/n_common_args__kernel_cycles/32/832/manual_time 59025480
BM_pgm_dispatch/n_common_args__kernel_cycles/32/896/manual_time 58741856
BM_pgm_dispatch/n_common_args__kernel_cycles/32/960/manual_time 60789114
BM_pgm_dispatch/n_common_args__kernel_cycles/32/1024/manual_time  60852770
BM_pgm_dispatch/n_common_args__kernel_cycles/64/0/manual_time 59210462
BM_pgm_dispatch/n_common_args__kernel_cycles/64/64/manual_time  64315175
BM_pgm_dispatch/n_common_args__kernel_cycles/64/128/manual_time 60844565
BM_pgm_dispatch/n_common_args__kernel_cycles/64/192/manual_time 60854019
BM_pgm_dispatch/n_common_args__kernel_cycles/64/256/manual_time 61829654
BM_pgm_dispatch/n_common_args__kernel_cycles/64/320/manual_time 61469529
BM_pgm_dispatch/n_common_args__kernel_cycles/64/384/manual_time 63622240
BM_pgm_dispatch/n_common_args__kernel_cycles/64/448/manual_time 61137457
BM_pgm_dispatch/n_common_args__kernel_cycles/64/512/manual_time 61330392
BM_pgm_dispatch/n_common_args__kernel_cycles/64/576/manual_time 60329014
BM_pgm_dispatch/n_common_args__kernel_cycles/64/640/manual_time 61248521
BM_pgm_dispatch/n_common_args__kernel_cycles/64/704/manual_time 61596842
BM_pgm_dispatch/n_common_args__kernel_cycles/64/768/manual_time 61041735
BM_pgm_dispatch/n_common_args__kernel_cycles/64/832/manual_time 60665798
BM_pgm_dispatch/n_common_args__kernel_cycles/64/896/manual_time 60693440
BM_pgm_dispatch/n_common_args__kernel_cycles/64/960/manual_time 61223041
BM_pgm_dispatch/n_common_args__kernel_cycles/64/1024/manual_time  63023408
BM_pgm_dispatch/n_common_args__kernel_cycles/96/0/manual_time 60143242
BM_pgm_dispatch/n_common_args__kernel_cycles/96/64/manual_time  59564053
BM_pgm_dispatch/n_common_args__kernel_cycles/96/128/manual_time 61034943
BM_pgm_dispatch/n_common_args__kernel_cycles/96/192/manual_time 61054441
BM_pgm_dispatch/n_common_args__kernel_cycles/96/256/manual_time 61140579
BM_pgm_dispatch/n_common_args__kernel_cycles/96/320/manual_time 60682771
BM_pgm_dispatch/n_common_args__kernel_cycles/96/384/manual_time 60923981
BM_pgm_dispatch/n_common_args__kernel_cycles/96/448/manual_time 61229457
BM_pgm_dispatch/n_common_args__kernel_cycles/96/512/manual_time 61091429
BM_pgm_dispatch/n_common_args__kernel_cycles/96/576/manual_time 58997882
BM_pgm_dispatch/n_common_args__kernel_cycles/96/640/manual_time 61230344
BM_pgm_dispatch/n_common_args__kernel_cycles/96/704/manual_time 61061651
BM_pgm_dispatch/n_common_args__kernel_cycles/96/768/manual_time 61616737
BM_pgm_dispatch/n_common_args__kernel_cycles/96/832/manual_time 60866592
BM_pgm_dispatch/n_common_args__kernel_cycles/96/896/manual_time 59085023
BM_pgm_dispatch/n_common_args__kernel_cycles/96/960/manual_time 60897409
BM_pgm_dispatch/n_common_args__kernel_cycles/96/1024/manual_time  61227681
BM_pgm_dispatch/n_common_args__kernel_cycles/128/0/manual_time  60801493
BM_pgm_dispatch/n_common_args__kernel_cycles/128/64/manual_time 58974744
BM_pgm_dispatch/n_common_args__kernel_cycles/128/128/manual_time  60764658
BM_pgm_dispatch/n_common_args__kernel_cycles/128/192/manual_time  60728525
BM_pgm_dispatch/n_common_args__kernel_cycles/128/256/manual_time  60771441
BM_pgm_dispatch/n_common_args__kernel_cycles/128/320/manual_time  58905012
BM_pgm_dispatch/n_common_args__kernel_cycles/128/384/manual_time  60779244
BM_pgm_dispatch/n_common_args__kernel_cycles/128/448/manual_time  59241541
BM_pgm_dispatch/n_common_args__kernel_cycles/128/512/manual_time  59220369
BM_pgm_dispatch/n_common_args__kernel_cycles/128/576/manual_time  61013887
BM_pgm_dispatch/n_common_args__kernel_cycles/128/640/manual_time  61019466
BM_pgm_dispatch/n_common_args__kernel_cycles/128/704/manual_time  60943575
BM_pgm_dispatch/n_common_args__kernel_cycles/128/768/manual_time  59557277
BM_pgm_dispatch/n_common_args__kernel_cycles/128/832/manual_time  61234976
BM_pgm_dispatch/n_common_args__kernel_cycles/128/896/manual_time  60323254
BM_pgm_dispatch/n_common_args__kernel_cycles/128/960/manual_time  59804196
BM_pgm_dispatch/n_common_args__kernel_cycles/128/1024/manual_time 60578111
BM_pgm_dispatch/n_common_args__kernel_cycles/160/0/manual_time  59548933
BM_pgm_dispatch/n_common_args__kernel_cycles/160/64/manual_time 59081174
BM_pgm_dispatch/n_common_args__kernel_cycles/160/128/manual_time  60418732
BM_pgm_dispatch/n_common_args__kernel_cycles/160/192/manual_time  58767184
BM_pgm_dispatch/n_common_args__kernel_cycles/160/256/manual_time  60975990
BM_pgm_dispatch/n_common_args__kernel_cycles/160/320/manual_time  61114888
BM_pgm_dispatch/n_common_args__kernel_cycles/160/384/manual_time  61637422
BM_pgm_dispatch/n_common_args__kernel_cycles/160/448/manual_time  60350815
BM_pgm_dispatch/n_common_args__kernel_cycles/160/512/manual_time  59972384
BM_pgm_dispatch/n_common_args__kernel_cycles/160/576/manual_time  58923891
BM_pgm_dispatch/n_common_args__kernel_cycles/160/640/manual_time  58882967
BM_pgm_dispatch/n_common_args__kernel_cycles/160/704/manual_time  61445052
BM_pgm_dispatch/n_common_args__kernel_cycles/160/768/manual_time  59079273
BM_pgm_dispatch/n_common_args__kernel_cycles/160/832/manual_time  58966666
BM_pgm_dispatch/n_common_args__kernel_cycles/160/896/manual_time  60919464
BM_pgm_dispatch/n_common_args__kernel_cycles/160/960/manual_time  61089230
BM_pgm_dispatch/n_common_args__kernel_cycles/160/1024/manual_time 60518569
BM_pgm_dispatch/n_common_args__kernel_cycles/192/0/manual_time  59122882
BM_pgm_dispatch/n_common_args__kernel_cycles/192/64/manual_time 60281460
BM_pgm_dispatch/n_common_args__kernel_cycles/192/128/manual_time  58688298
BM_pgm_dispatch/n_common_args__kernel_cycles/192/192/manual_time  58589310
BM_pgm_dispatch/n_common_args__kernel_cycles/192/256/manual_time  58681323
BM_pgm_dispatch/n_common_args__kernel_cycles/192/320/manual_time  59499118
BM_pgm_dispatch/n_common_args__kernel_cycles/192/384/manual_time  58641552
BM_pgm_dispatch/n_common_args__kernel_cycles/192/448/manual_time  60606680
BM_pgm_dispatch/n_common_args__kernel_cycles/192/512/manual_time  59084967
BM_pgm_dispatch/n_common_args__kernel_cycles/192/576/manual_time  59734723
BM_pgm_dispatch/n_common_args__kernel_cycles/192/640/manual_time  60118671
BM_pgm_dispatch/n_common_args__kernel_cycles/192/704/manual_time  58923715
BM_pgm_dispatch/n_common_args__kernel_cycles/192/768/manual_time  60533705
BM_pgm_dispatch/n_common_args__kernel_cycles/192/832/manual_time  58739601
BM_pgm_dispatch/n_common_args__kernel_cycles/192/896/manual_time  58657990
BM_pgm_dispatch/n_common_args__kernel_cycles/192/960/manual_time  58991618
BM_pgm_dispatch/n_common_args__kernel_cycles/192/1024/manual_time 58892440
BM_pgm_dispatch/n_common_args__kernel_cycles/224/0/manual_time  60096772
BM_pgm_dispatch/n_common_args__kernel_cycles/224/64/manual_time 59800673
BM_pgm_dispatch/n_common_args__kernel_cycles/224/128/manual_time  59721196
BM_pgm_dispatch/n_common_args__kernel_cycles/224/192/manual_time  58874768
BM_pgm_dispatch/n_common_args__kernel_cycles/224/256/manual_time  60852772
BM_pgm_dispatch/n_common_args__kernel_cycles/224/320/manual_time  62361935
BM_pgm_dispatch/n_common_args__kernel_cycles/224/384/manual_time  60017594
BM_pgm_dispatch/n_common_args__kernel_cycles/224/448/manual_time  60450282
BM_pgm_dispatch/n_common_args__kernel_cycles/224/512/manual_time  60900480
BM_pgm_dispatch/n_common_args__kernel_cycles/224/576/manual_time  62711744
BM_pgm_dispatch/n_common_args__kernel_cycles/224/640/manual_time  61796612
BM_pgm_dispatch/n_common_args__kernel_cycles/224/704/manual_time  60729482
BM_pgm_dispatch/n_common_args__kernel_cycles/224/768/manual_time  60820027
BM_pgm_dispatch/n_common_args__kernel_cycles/224/832/manual_time  59852456
BM_pgm_dispatch/n_common_args__kernel_cycles/224/896/manual_time  60306128
BM_pgm_dispatch/n_common_args__kernel_cycles/224/960/manual_time  59120996
BM_pgm_dispatch/n_common_args__kernel_cycles/224/1024/manual_time 60551761
BM_pgm_dispatch/n_common_args__kernel_cycles/256/0/manual_time  58939821
BM_pgm_dispatch/n_common_args__kernel_cycles/256/64/manual_time 59192915
BM_pgm_dispatch/n_common_args__kernel_cycles/256/128/manual_time  58880812
BM_pgm_dispatch/n_common_args__kernel_cycles/256/192/manual_time  58569891
BM_pgm_dispatch/n_common_args__kernel_cycles/256/256/manual_time  58913198
BM_pgm_dispatch/n_common_args__kernel_cycles/256/320/manual_time  58825842
BM_pgm_dispatch/n_common_args__kernel_cycles/256/384/manual_time  60805988
BM_pgm_dispatch/n_common_args__kernel_cycles/256/448/manual_time  58625367
BM_pgm_dispatch/n_common_args__kernel_cycles/256/512/manual_time  58880546
BM_pgm_dispatch/n_common_args__kernel_cycles/256/576/manual_time  60997121
BM_pgm_dispatch/n_common_args__kernel_cycles/256/640/manual_time  59022349
BM_pgm_dispatch/n_common_args__kernel_cycles/256/704/manual_time  58643833
BM_pgm_dispatch/n_common_args__kernel_cycles/256/768/manual_time  60305653
BM_pgm_dispatch/n_common_args__kernel_cycles/256/832/manual_time  60222695
BM_pgm_dispatch/n_common_args__kernel_cycles/256/896/manual_time  60723350
BM_pgm_dispatch/n_common_args__kernel_cycles/256/960/manual_time  58897849
BM_pgm_dispatch/n_common_args__kernel_cycles/256/1024/manual_time 58967906
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


# Compute average time per n_common_args
avg_df = df.groupby("n_common_args", as_index=False)["time_ns"].mean()

# Export the average data to CSV
avg_df.to_csv("benchmark_avg_results.csv", index=False)

# Plotting average time
plt.figure(figsize=(8, 5))
sns.barplot(data=avg_df, x="n_common_args", y="time_ns", palette="tab10")
plt.ylim(avg_df["time_ns"].min() * 0.995, avg_df["time_ns"].max() * 1.005)

plt.title("Average Benchmark Time per n_common_args")
plt.xlabel("n_common_args")
plt.ylabel("Average Time (ns)")
plt.grid(axis="y")
plt.tight_layout()

# Save average plot to PNG
plt.savefig("benchmark_avg_plot.png", dpi=300)
plt.show()
