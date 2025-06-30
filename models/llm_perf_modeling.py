from math import ceil
import pandas as pd
import argparse


class Chip:
    def __init__(
        self, name, peak_memory_bandwidth_gb, flops, freq, memory_capacity_gb, memory_efficiency, compute_efficiency
    ) -> None:
        self.name = name
        self.peak_memory_bandwidth_gb = peak_memory_bandwidth_gb
        self.flops = flops
        self.freq = freq
        self.memory_capacity_gb = memory_capacity_gb
        self.memory_efficiency = memory_efficiency
        self.compute_efficiency = compute_efficiency
        self.effective_gflops = self.flops * self.freq * self.compute_efficiency / 1e9
        self.effective_memory_bandwidth_GBps = self.peak_memory_bandwidth_gb * self.memory_efficiency

    def print(self):
        print(
            f"  {self.name}:\n"
            f"\tpeak_memory_bandwidth:{self.peak_memory_bandwidth_gb} GB/s\n"
            f"\tflops:{self.flops} GFLOPs\n"
            f"\tfreq:{self.freq} Hz\n"
            f"\tmemory_capacity:{self.memory_capacity_gb} GB\n"
            f"\tmemory_efficiency:{self.memory_efficiency}\n"
            f"\tcompute_efficiency:{self.compute_efficiency}\n"
            f"\teffective_gflops:{self.effective_gflops} GFLOPs\n"
            f"\teffective_memory_bandwidth_GBps:{self.effective_memory_bandwidth_GBps} GB/s\n"
        )


class System:
    def __init__(self, name, chip, num_instances) -> None:
        self.name = name
        self.chip = chip
        self.num_instances = num_instances
        self.effective_gflops = self.chip.effective_gflops * self.num_instances
        self.effective_memory_bandwidth_GBps = self.chip.effective_memory_bandwidth_GBps * self.num_instances
        self.memory_capacity_gb = self.chip.memory_capacity_gb * self.num_instances

    def print(self):
        print(
            f"  {self.name}:\n"
            f"\tnum_instances:{self.num_instances}\n"
            f"\tchip:{self.chip.name}\n"
            f"\tcapacity:{self.chip.memory_capacity_gb} GB\n"
            f"\tflops:{self.effective_gflops} GFLOPs\n"
            f"\tbandwidth:{self.effective_memory_bandwidth_GBps} GB/s\n"
        )


class TransformerModel:
    def __init__(
        self,
        name,
        num_parameters_B,
        input_sequence_length,
        output_sequence_length,
        num_layers,
        hidden_size,
        num_q_heads,
        num_kv_heads,
        intermediate_size,
        vocab_size,
    ) -> None:
        self.name = name
        self.num_parameters_B = num_parameters_B
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.head_dim = hidden_size // num_q_heads
        self.average_sequence_length = input_sequence_length + output_sequence_length / 2
        self.total_sequence_length = input_sequence_length + output_sequence_length

    # TODO: move inside calculate
    # helper functions
    def dram_loading_mm_compute(self, row_size):
        return self.num_parameters_B * row_size * 2

    def attention_mm_compute(self, row_size, col_size):
        return self.num_layers * self.num_q_heads * self.head_dim * row_size * col_size * 2 * 2 / 1024**3

    def calculate_per_user(self, system):
        estimates_per_user = {}

        estimates_per_user["num_parameters(B)"] = self.num_parameters_B

        ################
        # model_size_B #
        ################
        attention_params = (
            self.hidden_size * self.hidden_size * ((self.num_q_heads + self.num_kv_heads) / self.num_q_heads + 1)
        )
        linear_params = self.hidden_size * self.hidden_size
        embedding_params = self.hidden_size * self.vocab_size
        if self.name.startswith("llama2") or self.name.startswith("llama3"):
            mlp_params = 3 * self.hidden_size * self.intermediate_size
            estimates_per_user["model_size(B)"] = (
                self.num_layers * (attention_params + linear_params + mlp_params) + embedding_params
            ) / 1e9
        else:
            estimates_per_user["model_size(B)"] = 0

        #################################
        # max_kv_cache_size_per_user(GB) #
        #################################
        estimates_per_user["max_kv_cache_size_per_user(GB)"] = (
            self.num_layers * self.total_sequence_length * self.num_kv_heads * self.head_dim * 2 / 1024**3
        )

        ############################
        # avg_kv_cache_size_per_user(GB) #
        ############################
        estimates_per_user["avg_kv_cache_size_per_user(GB)"] = (
            self.num_layers * self.average_sequence_length * self.num_kv_heads * self.head_dim * 2 / 1024**3
        )

        #################################
        # max_num_users_that_fit_in_memory #
        #################################
        if system.memory_capacity_gb < self.num_parameters_B:
            estimates_per_user["max_num_users_that_fit_in_memory"] = 0
        else:
            estimates_per_user["max_num_users_that_fit_in_memory"] = (
                system.memory_capacity_gb - self.num_parameters_B
            ) // estimates_per_user["max_kv_cache_size_per_user(GB)"]

        ############################
        # prefill_compute(GFLOPS) #
        ############################

        print(f"prefill_dram_loading_mm_compute: {self.dram_loading_mm_compute(self.input_sequence_length)}")
        print(f"prefill_attention_mm_compute: {self.attention_mm_compute(self.input_sequence_length, self.input_sequence_length)}")
        estimates_per_user["prefill_compute(GFLOPS)"] = self.dram_loading_mm_compute(
            self.input_sequence_length
        ) + self.attention_mm_compute(self.input_sequence_length, self.input_sequence_length)

        print(f"prefill_compute(GFLOPS): {estimates_per_user['prefill_compute(GFLOPS)']}")
        print(f"system.effective_gflops: {system.effective_gflops}")

        ############################
        # prefill_compute_latency(ms) #
        ############################
        estimates_per_user["prefill_compute_latency(ms)"] = (
            estimates_per_user["prefill_compute(GFLOPS)"] / system.effective_gflops * 1000
        )

        return estimates_per_user

    def calculate_per_batch(self, estimates_per_user, num_users, system):
        estimates_per_batch = {}

        ############################
        # max_kv_cache_size(GB) #
        ############################
        estimates_per_batch["max_kv_cache_size(GB)"] = estimates_per_user["max_kv_cache_size_per_user(GB)"] * num_users

        ############################
        # max_memory_size(GB) #
        ############################
        estimates_per_batch["max_memory_size(GB)"] = (
            estimates_per_batch["max_kv_cache_size(GB)"] + self.num_parameters_B
        )

        ############################
        # avg_kv_cache_size(GB) #
        ############################
        estimates_per_batch["avg_kv_cache_size(GB)"] = estimates_per_user["avg_kv_cache_size_per_user(GB)"] * num_users

        ############################
        # avg_memory_size(GB) #
        ############################
        estimates_per_batch["avg_memory_size(GB)"] = (
            estimates_per_batch["avg_kv_cache_size(GB)"] + self.num_parameters_B
        )

        ############################
        # decode_compute(GFLOPS) #
        ############################
        # old version:
        # estimates_per_batch["decode_compute(GFLOPS)"] = self.avg_memory_size_GB(num_users=1) * 2 * ceil(num_users / 32) * 32
        # new version:
        estimates_per_batch["decode_compute(GFLOPS)"] = self.dram_loading_mm_compute(
            ceil(num_users / 32) * 32
        ) + self.attention_mm_compute(num_users, self.average_sequence_length)

        ############################
        # decode_compute_latency(ms) #
        ############################
        if estimates_per_user["max_num_users_that_fit_in_memory"] < num_users:
            estimates_per_batch["decode_compute_latency(ms)"] = -1
        else:
            estimates_per_batch["decode_compute_latency(ms)"] = (
                estimates_per_batch["decode_compute(GFLOPS)"] / system.effective_gflops * 1000
            )

        ############################
        # decode_memory_latency(ms) #
        ############################
        if estimates_per_user["max_num_users_that_fit_in_memory"] < num_users:
            estimates_per_batch["decode_memory_latency(ms)"] = -1
        else:
            estimates_per_batch["decode_memory_latency(ms)"] = (
                estimates_per_batch["avg_memory_size(GB)"] / system.effective_memory_bandwidth_GBps * 1000
            )

        ############################
        # decode_latency(ms) #
        ############################
        if estimates_per_user["max_num_users_that_fit_in_memory"] < num_users:
            estimates_per_batch["decode_latency(ms)"] = -1
        else:
            estimates_per_batch["decode_latency(ms)"] = max(
                estimates_per_batch["decode_compute_latency(ms)"], estimates_per_batch["decode_memory_latency(ms)"]
            )

        ############################
        # decode_throughput(t/s/u) #
        ############################
        if estimates_per_user["max_num_users_that_fit_in_memory"] < num_users:
            estimates_per_batch["decode_throughput(t/s/u)"] = -1
        else:
            estimates_per_batch["decode_throughput(t/s/u)"] = 1000 / estimates_per_batch["decode_latency(ms)"]

        ############################
        # decode_throughput(t/s) #
        ############################
        if estimates_per_user["max_num_users_that_fit_in_memory"] < num_users:
            estimates_per_batch["decode_throughput(t/s)"] = -1
        else:
            estimates_per_batch["decode_throughput(t/s)"] = estimates_per_batch["decode_throughput(t/s/u)"] * num_users

        ############################
        # decode_total_time(ms) #
        ############################
        if estimates_per_user["max_num_users_that_fit_in_memory"] < num_users:
            estimates_per_batch["decode_total_time(ms)"] = -1
        else:
            estimates_per_batch["decode_total_time(ms)"] = (
                estimates_per_batch["decode_latency(ms)"] * self.output_sequence_length
            )

        ############################
        # prefill_memory_latency(ms) #
        ############################
        if estimates_per_user["max_num_users_that_fit_in_memory"] < num_users:
            estimates_per_batch["prefill_memory_latency(ms)"] = -1
        else:
            estimates_per_batch["prefill_memory_latency(ms)"] = estimates_per_batch["decode_memory_latency(ms)"]

        ############################
        # prefill_latency(ms) #
        ############################
        if estimates_per_user["max_num_users_that_fit_in_memory"] < num_users:
            estimates_per_batch["prefill_latency(ms)"] = -1
        else:
            estimates_per_batch["prefill_latency(ms)"] = max(
                estimates_per_user["prefill_compute_latency(ms)"], estimates_per_batch["prefill_memory_latency(ms)"]
            )

        ############################
        # prefill_throughput(u/s) #
        ############################
        if estimates_per_user["max_num_users_that_fit_in_memory"] < num_users:
            estimates_per_batch["prefill_throughput(u/s)"] = -1
        else:
            estimates_per_batch["prefill_throughput(u/s)"] = 1000 / estimates_per_batch["prefill_latency(ms)"]

        ############################
        # prefill_throughput(t/s) #
        ############################
        if estimates_per_user["max_num_users_that_fit_in_memory"] < num_users:
            estimates_per_batch["prefill_throughput(t/s)"] = -1
        else:
            estimates_per_batch["prefill_throughput(t/s)"] = (
                estimates_per_batch["prefill_throughput(u/s)"] * self.input_sequence_length
            )

        ############################
        # prefill_total_time(ms) #
        ############################
        if estimates_per_user["max_num_users_that_fit_in_memory"] < num_users:
            estimates_per_batch["prefill_total_time(ms)"] = -1
        else:
            estimates_per_batch["prefill_total_time(ms)"] = estimates_per_batch["prefill_latency(ms)"] * num_users

        ############################
        # time_to_first_token(ms) #
        ############################
        if estimates_per_user["max_num_users_that_fit_in_memory"] < num_users:
            estimates_per_batch["time_to_first_token(ms)"] = -1
        else:
            estimates_per_batch["time_to_first_token(ms)"] = estimates_per_batch["prefill_latency(ms)"]

        ############################
        # time_to_last_token(ms) #
        ############################
        if estimates_per_user["max_num_users_that_fit_in_memory"] < num_users:
            estimates_per_batch["time_to_last_token(ms)"] = -1
        else:
            estimates_per_batch["time_to_last_token(ms)"] = (
                estimates_per_batch["prefill_total_time(ms)"] + estimates_per_batch["decode_total_time(ms)"]
            )

        ############################
        # overall_throughput(t/s/u) #
        ############################
        if estimates_per_user["max_num_users_that_fit_in_memory"] < num_users:
            estimates_per_batch["overall_throughput(t/s/u)"] = -1
        else:
            estimates_per_batch["overall_throughput(t/s/u)"] = (
                self.output_sequence_length * 1000 / estimates_per_batch["time_to_last_token(ms)"]
            )

        ############################
        # overall_throughput(t/s) #
        ############################
        if estimates_per_user["max_num_users_that_fit_in_memory"] < num_users:
            estimates_per_batch["overall_throughput(t/s)"] = -1
        else:
            estimates_per_batch["overall_throughput(t/s)"] = (
                estimates_per_batch["overall_throughput(t/s/u)"] * num_users
            )

        return estimates_per_batch

    def calculate_misc(self, estimates_per_user, system):
        estimates_misc = {}
        estimates_per_batch = self.calculate_per_batch(
            estimates_per_user, estimates_per_user["max_num_users_that_fit_in_memory"], system
        )
        estimates_misc["overall_throughput_at_max_num_users(t/s)"] = estimates_per_batch["overall_throughput(t/s)"]
        return estimates_misc

    def calculate(self, num_users, system):
        estimates_per_user = self.calculate_per_user(system)
        estimates_per_batch = self.calculate_per_batch(estimates_per_user, num_users, system)
        estimates_misc = self.calculate_misc(estimates_per_user, system)
        # merge dictionaries
        return {**estimates_per_user, **estimates_per_batch, **estimates_misc}

    def set_sequence_length(self, input_sequence_length, output_sequence_length):
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.average_sequence_length = input_sequence_length + output_sequence_length / 2
        self.total_sequence_length = input_sequence_length + output_sequence_length


def print_table(metric, column_names, row_names, table):
    # pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 1000)
    # pd.set_option("display.large_repr", 'info')
    # pd.set_option("display.expand_frame_repr", True)
    pd.set_option("display.width", 10000)
    pd.set_option("display.max_colwidth", 1000)
    # pd.set_option("display.precision", 5)

    print("=======================")
    print(f"{metric}")
    print("=======================")
    df = pd.DataFrame(table, columns=column_names, index=row_names)
    print(df)


# convert
# estimates[model_name][system_name][input_sequence_length][output_sequence_length][metric] = value
# to
# new_estimates[metric][system_name][model_name][input_sequence_length+output_sequence_length] = value
def convert_estimates_layout(estimates):
    new_estimates = {}
    for model_name, systems in estimates.items():
        for system_name, sequence_lengths in systems.items():
            for input_sequence_length, output_sequence_lengths in sequence_lengths.items():
                for output_sequence_length, estimates in output_sequence_lengths.items():
                    sequence_length = str(input_sequence_length) + "+" + str(output_sequence_length)
                    for metric, value in estimates.items():
                        if metric not in new_estimates:
                            new_estimates[metric] = {}
                        if system_name not in new_estimates[metric]:
                            new_estimates[metric][system_name] = {}
                        if model_name not in new_estimates[metric][system_name]:
                            new_estimates[metric][system_name][model_name] = {}
                        assert sequence_length not in new_estimates[metric][system_name][model_name]
                        new_estimates[metric][system_name][model_name][sequence_length] = value
    return new_estimates


def print_estimates(num_users, estimates, estimates_to_print=set()):
    print("+++++++++++++++++++++")
    print(f"+++ num_users: {num_users} +++")
    print("+++++++++++++++++++++")
    for metric, systems in estimates.items():
        if len(estimates_to_print) > 0 and metric not in estimates_to_print:
            continue
        column_names = []
        row_names = []
        rows = {}
        for system_name, models in systems.items():
            for model_name, sequence_lengths in models.items():
                column_names.append(system_name + "_" + model_name)
                for sequence_length, value in sequence_lengths.items():
                    if sequence_length not in rows:
                        row_names.append(sequence_length)
                        rows[sequence_length] = []
                    rows[sequence_length].append(value)
        table = []
        for sequence_length in row_names:
            table.append(rows[sequence_length])

        print_table(metric, column_names, row_names, table)


def create_chips():
    chips = {}
    chips["WH"] = Chip(
        name="WH",
        peak_memory_bandwidth_gb=336,
        flops=8 * 16 * 16 * 64 * 2,
        freq=1e9,
        memory_capacity_gb=12,
        memory_efficiency=0.8928,
        # compute_efficiency=0.60
        compute_efficiency=1,
        # memory_efficiency=1.19,
        # compute_efficiency=0.8,
    )

    chips["BH"] = Chip(
        name="BH",
        peak_memory_bandwidth_gb=512,
        flops=8 * 16 * 16 * 140 * 2,
        freq=1e9,
        memory_capacity_gb=32,
        # memory_efficiency=0.75,
        # compute_efficiency=0.60
        memory_efficiency=1,
        compute_efficiency=0.8,
    )

    chips["H100"] = Chip(
        name="H100",
        peak_memory_bandwidth_gb=3.35 * 1024,
        flops=2000000,
        freq=1e9,
        memory_capacity_gb=80,
        memory_efficiency=1,
        compute_efficiency=1,
    )

    chips["H200"] = Chip(
        name="H200",
        peak_memory_bandwidth_gb=4.8 * 1024,
        flops=2000000,
        freq=1e9,
        memory_capacity_gb=141,
        memory_efficiency=1,
        compute_efficiency=1,
    )

    return chips


def create_systems(chips):
    systems = {}
    systems["WH_N150"] = System(name="WH_N150", chip=chips["WH"], num_instances=1)
    systems["WH_N300"] = System(name="WH_N300", chip=chips["WH"], num_instances=2)
    systems["WH_Galaxy_x1"] = System(name="WH_Galaxy_x1", chip=chips["WH"], num_instances=32)
    systems["WH_Galaxy_x2"] = System(name="WH_Galaxy_x2", chip=chips["WH"], num_instances=64)
    systems["WH_Galaxy_x4"] = System(name="WH_Galaxy_x4", chip=chips["WH"], num_instances=128)
    systems["BH_Galaxy_x1"] = System(name="BH_Galaxy_x1", chip=chips["BH"], num_instances=32)
    systems["BH_Galaxy_x2"] = System(name="BH_Galaxy_x2", chip=chips["BH"], num_instances=64)
    systems["BH_Galaxy_x3"] = System(name="BH_Galaxy_x3", chip=chips["BH"], num_instances=96)
    systems["BH_Galaxy_x4"] = System(name="BH_Galaxy_x4", chip=chips["BH"], num_instances=128)
    systems["BH_Galaxy_x6"] = System(name="BH_Galaxy_x6", chip=chips["BH"], num_instances=192)
    systems["H100_x1"] = System(name="H100_x1", chip=chips["H100"], num_instances=1)
    systems["H100_x4"] = System(name="H100_x4", chip=chips["H100"], num_instances=4)
    systems["H100_DGX"] = System(name="H100_DGX", chip=chips["H100"], num_instances=8)
    systems["H200_DGX"] = System(name="H200_DGX", chip=chips["H200"], num_instances=8)
    return systems


def create_models():
    models = {}
    models["llama3_8B"] = TransformerModel(
        name="llama3_8B",
        num_parameters_B=8,
        input_sequence_length=1024,
        output_sequence_length=1024,
        num_layers=32,
        hidden_size=4096,
        num_q_heads=32,
        num_kv_heads=8,
        intermediate_size=14336,
        vocab_size=128256,
    )

    models["llama3_3B"] = TransformerModel(
        name="llama3_3B",
        num_parameters_B=3,
        input_sequence_length=1024,
        output_sequence_length=1024,
        num_layers=28,
        hidden_size=3072,
        num_q_heads=24,
        num_kv_heads=8,
        intermediate_size=8192,
        vocab_size=128256,
    )

    models["llama3_70B"] = TransformerModel(
        name="llama3_70B",
        num_parameters_B=70,
        input_sequence_length=1024,
        output_sequence_length=1024,
        num_layers=80,
        hidden_size=8192,
        num_q_heads=64,
        num_kv_heads=8,
        intermediate_size=28672,
        vocab_size=128256,
    )

    # old estimated model parameters used for some customer slides
    models["llama3_400B"] = TransformerModel(
        name="llama3_400B",
        num_parameters_B=400,
        input_sequence_length=1024,
        output_sequence_length=1024,
        num_layers=105,
        hidden_size=16384,
        num_q_heads=128,
        num_kv_heads=16,
        intermediate_size=65536,
        vocab_size=128256,
    )

    models["llama3_405B"] = TransformerModel(
        name="llama3_405B",
        num_parameters_B=405,
        input_sequence_length=2048,
        output_sequence_length=128,
        num_layers=126,
        hidden_size=16384,
        num_q_heads=128,
        num_kv_heads=8,
        intermediate_size=53248,
        vocab_size=128256,
    )

    models["llama3_212B"] = TransformerModel(
        name="llama3_212B",
        num_parameters_B=212,
        input_sequence_length=1024,
        output_sequence_length=1024,
        num_layers=80,
        hidden_size=16384,
        num_q_heads=64,
        num_kv_heads=8,
        intermediate_size=40960,
        vocab_size=128256,
    )

    models["llama3_1TB"] = TransformerModel(
        name="llama3_1TB",
        num_parameters_B=1024,
        input_sequence_length=1024,
        output_sequence_length=1024,
        num_layers=100,
        hidden_size=32768,
        num_q_heads=64,
        num_kv_heads=8,
        intermediate_size=81920,
        vocab_size=128256,
    )

    return models


def main():
    chips_db = create_chips()
    systems_db = create_systems(chips_db)
    models_db = create_models()

    # create argument parser
    parser = argparse.ArgumentParser(description="Performance Modeling")

    # add arguments
    # parser.add_argument("-h", "--help", help="Show this help message and exit")
    parser.add_argument("-m", "--models", nargs="+", help="List of model names")
    parser.add_argument("-s", "--systems", nargs="+", help="List of system names")
    parser.add_argument("-u", "--num_users", type=int, help="Number of users")
    parser.add_argument("-i", "--input_length", type=int, help="Input sequence length")
    parser.add_argument("-o", "--output_length", type=int, help="Output sequence length")
    parser.add_argument("-p", "--print_all", help="Print all configs", action="store_true")

    # parse arguments
    args = parser.parse_args()

    # access the parsed arguments
    models = args.models
    systems = args.systems
    num_users = args.num_users
    input_length = args.input_length
    output_length = args.output_length
    print_all = args.print_all

    if print_all:
        print("chips:")
        for _, chip in chips_db.items():
            chip.print()
        print("systems:")
        for _, system in systems_db.items():
            system.print()
        print("models:")
        for _, model in models_db.items():
            print(f"  {model.name}")
        return

    if models is None or systems is None or num_users is None or input_length is None or output_length is None:
        print("Please provide all arguments")
        return

    # if args.help:
    #     parser.print_help()
    #     return

    # TODO: Use the parsed arguments in your code

    # TODO:
    # 2. remove sequence length from the model, add it to the calculation function
    # 3. add max_overall_thoughput_at_some_batch function (overall throughput drops sometimes due to prefill)

    # Build database for the performnace data
    estimates = {}
    # for system in [WH_Galaxy_x1, BH_Galaxy_x1, BH_Galaxy_x2, BH_Galaxy_x3, BH_Galaxy_x4, BH_Galaxy_x6]:
    # for system in [WH_Galaxy_x1, WH_Galaxy_x4]:
    for system in [systems_db[system_name] for system_name in systems]:
        # for model in [llama3_70B, llama3_212B, llama3_1TB]:
        for model in [models_db[model_name] for model_name in models]:
            # for input_sequence_length in [100, 1024, 7*1024, 31*1024, 199*1024]:
            for input_sequence_length in [input_length]:
                # output_sequence_length = 1024 if input_sequence_length > 100 else 100
                for output_sequence_length in [output_length]:
                    model.set_sequence_length(input_sequence_length, output_sequence_length)
                    if model.name not in estimates:
                        estimates[model.name] = {}
                    if system.name not in estimates[model.name]:
                        estimates[model.name][system.name] = {}
                    if input_sequence_length not in estimates[model.name][system.name]:
                        estimates[model.name][system.name][input_sequence_length] = {}
                    estimates[model.name][system.name][input_sequence_length][output_sequence_length] = model.calculate(
                        num_users, system
                    )

    new_estimates = convert_estimates_layout(estimates)

    # Print the estimates data to stdout
    print_estimates(
        num_users,
        new_estimates,
        estimates_to_print={
            "max_kv_cache_size_per_user(GB)",
            "decode_compute_latency(ms)",
            "decode_memory_latency(ms)",
            "decode_latency(ms)",
            "decode_throughput(t/s/u)",
            "decode_throughput(t/s)",
            "time_to_first_token(ms)",
            # 'time_to_last_token(ms)',
            "overall_throughput(t/s/u)",
            "overall_throughput(t/s)",
            "max_num_users_that_fit_in_memory",
            # 'overall_throughput_at_max_num_users(t/s)'
        },
    )


if __name__ == "__main__":
    main()
