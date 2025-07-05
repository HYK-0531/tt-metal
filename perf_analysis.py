import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys


def load_performance_data(json_file):
    """Load performance metrics from JSON file"""
    with open(json_file, "r") as f:
        return json.load(f)


def analyze_time_breakdown(data):
    """Analyze and break down time spending according to the structure"""

    # Extract raw metrics
    preprocess_time = data["preprocess_time"]
    vision_model_prefill_time = data["vision_model_prefill_time"]
    compile_prefill_time = data["compile_prefill_time"]
    inference_prefill_time = data["inference_prefill_time"]
    preprocess_decode_time = data["preprocess_decode_time"]
    inference_decode_time = data["inference_decode_time"]
    compile_decode_time = data["compile_decode_time"]

    # Calculate breakdown according to user specifications
    # vision_model_prefill_time is part of preprocess_time
    other_preprocess_time = preprocess_time - vision_model_prefill_time

    # Break down inference_decode_time into components
    sum_individual_decode_times = sum(data["inference_decode_times"])
    other_decode_time = inference_decode_time - compile_decode_time - sum_individual_decode_times

    # Create time breakdown dictionary
    time_breakdown = {
        "Vision Model Prefill": vision_model_prefill_time,
        "Other Preprocessing": other_preprocess_time,
        "Compile Prefill": compile_prefill_time,
        "Inference Prefill": inference_prefill_time,
        "Preprocess Decode": preprocess_decode_time,
        "Compile Decode": compile_decode_time,
        "Actual Inference Decode": sum_individual_decode_times,
        "Other Decode Time": other_decode_time,
    }

    return time_breakdown


def create_pie_chart(time_breakdown, data):
    """Create a pie chart showing time breakdown"""

    # Prepare data for pie chart
    labels = list(time_breakdown.keys())
    sizes = list(time_breakdown.values())
    total_time = sum(sizes)

    # Define colors for different phases
    colors = [
        "#FF6B6B",  # Vision Model Prefill - Red
        "#FFB347",  # Other Preprocessing - Orange
        "#87CEEB",  # Compile Prefill - Sky Blue
        "#98FB98",  # Inference Prefill - Pale Green
        "#DDA0DD",  # Preprocess Decode - Plum
        "#4169E1",  # Compile Decode - Royal Blue
        "#32CD32",  # Actual Inference Decode - Lime Green
        "#FFA500",  # Other Decode Time - Dark Orange
    ]

    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Create explode array - slightly explode small slices to make them visible
    explode = []
    for size in sizes:
        if size / sum(sizes) < 0.02:  # Small slices (< 2%)
            explode.append(0.1)  # Explode small slices
        else:
            explode.append(0)

    # Custom autopct function to hide percentages for tiny slices to avoid overlap
    def autopct_func(pct):
        return f"{pct:.1f}%" if pct > 1.0 else ""

    # Main pie chart with exploded small slices
    wedges, texts, autotexts = ax1.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=autopct_func,
        startangle=90,
        explode=explode,
        textprops={"fontsize": 9},
        labeldistance=1.15,
        pctdistance=0.8,
        wedgeprops=dict(edgecolor="white", linewidth=1),
    )

    # Use matplotlib's automatic label positioning to avoid overlaps
    # Manually position problematic small labels to prevent overlap
    for i, text in enumerate(texts):
        if sizes[i] / sum(sizes) < 0.01:  # Very small slices (< 1%)
            text.set_fontsize(8)
            # Position the 0.0% slice label far away to avoid any overlap
            if labels[i] == "Preprocess Decode":
                text.set_position((1.5, -0.6))  # Move further down and right
                # Add arrow pointing to the tiny slice
                ax1.annotate(
                    "", xy=(0.05, -0.1), xytext=(1.4, -0.55), arrowprops=dict(arrowstyle="->", color="gray", lw=0.8)
                )
        elif sizes[i] / sum(sizes) < 0.02:  # Small slices (< 2%) like Inference Prefill
            text.set_fontsize(8)
            if labels[i] == "Inference Prefill":
                # Move this one to the left side to separate from Compile Decode
                text.set_position((-1.4, 0.2))
                # Add arrow pointing to slice
                ax1.annotate(
                    "", xy=(-0.1, 0.05), xytext=(-1.3, 0.15), arrowprops=dict(arrowstyle="->", color="gray", lw=0.8)
                )

    # Get model name from data, with fallback
    model_name = data.get("model_name", "Unknown Model")

    ax1.set_title(
        f"{model_name} Performance Breakdown\n"
        f'Device: {data["device_name"]} | Batch Size: {data["batch_size"]}\n'
        f"Total Time: {total_time:.2f}s",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Create detailed breakdown table
    ax2.axis("off")
    table_data = []
    table_data.append(["Component", "Time (s)", "Percentage", "Details"])
    table_data.append(["", "", "", ""])

    for i, (label, time_val) in enumerate(time_breakdown.items()):
        percentage = (time_val / total_time) * 100

        # Add context for each component
        details = ""
        if "Vision" in label:
            details = "Vision encoder processing"
        elif "Other Preprocessing" in label:
            details = "Text tokenization, etc."
        elif "Compile Prefill" in label:
            details = "Model compilation (prefill)"
        elif "Inference Prefill" in label:
            details = "First token generation"
        elif "Preprocess Decode" in label:
            details = "Decode preprocessing"
        elif "Compile Decode" in label:
            details = "Model compilation (decode)"
        elif "Actual Inference" in label:
            decode_tokens = len(data["inference_decode_times"])
            # Use actual individual decode times for accurate per-token calculation
            assert decode_tokens > 0, "Error: No decode tokens found"
            avg_per_token = time_val / decode_tokens
            details = f"{decode_tokens} tokens, {avg_per_token*1000:.1f}ms/token"
        elif "Other Decode" in label:
            details = "Decode framework overhead"

        table_data.append([label, f"{time_val:.3f}", f"{percentage:.1f}%", details])

    # Add total row
    table_data.append(["", "", "", ""])
    table_data.append(["TOTAL", f"{total_time:.3f}", "100.0%", ""])

    table = ax2.table(cellText=table_data, cellLoc="left", loc="center", colWidths=[0.35, 0.15, 0.15, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style the table with grouped colors
    for i in range(len(table_data)):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor("#4472C4")
                cell.set_text_props(weight="bold", color="white")
            elif i == len(table_data) - 1:  # Total row
                cell.set_facecolor("#E7E6E6")
                cell.set_text_props(weight="bold")
            elif table_data[i][0] == "":  # Empty rows
                cell.set_facecolor("#FFFFFF")
            else:
                # Group components by phase with consistent colors
                component_name = table_data[i][0]
                if component_name in ["Vision Model Prefill", "Other Preprocessing"]:
                    cell.set_facecolor("#FFE6E6")  # Light red for preprocessing
                elif component_name in ["Compile Prefill", "Inference Prefill"]:
                    cell.set_facecolor("#E6F3FF")  # Light blue for prefill
                elif component_name in [
                    "Preprocess Decode",
                    "Compile Decode",
                    "Actual Inference Decode",
                    "Other Decode Time",
                ]:
                    cell.set_facecolor("#E6FFE6")  # Light green for decode
                else:
                    cell.set_facecolor("#F2F2F2")  # Default gray

    ax2.set_title("Detailed Breakdown", fontsize=12, fontweight="bold")

    plt.tight_layout()
    return fig


def print_analysis_summary(time_breakdown, data):
    """Print detailed analysis summary"""
    total_time = sum(time_breakdown.values())
    decode_tokens = len(data["inference_decode_times"])

    # Get model name from data, with fallback
    model_name = data.get("model_name", "Unknown Model")

    print("=" * 60)
    print(f"{model_name.upper()} PERFORMANCE ANALYSIS")
    print("=" * 60)
    print(f"Device: {data['device_name']}")
    print(f"Batch Size: {data['batch_size']}")
    print(f"Total Runtime: {total_time:.3f} seconds")
    print()

    print("TIME BREAKDOWN:")
    print("-" * 40)
    for component, time_val in time_breakdown.items():
        percentage = (time_val / total_time) * 100
        print(f"{component:25}: {time_val:7.3f}s ({percentage:5.1f}%)")

    print()
    print("KEY INSIGHTS:")
    print("-" * 40)

    # Compilation vs inference analysis
    compile_time = time_breakdown["Compile Prefill"] + time_breakdown["Compile Decode"]
    inference_time = (
        time_breakdown["Actual Inference Decode"]
        + time_breakdown["Inference Prefill"]
        + time_breakdown["Other Decode Time"]
    )

    print(f"• Total Compilation Time: {compile_time:.3f}s ({compile_time/total_time*100:.1f}%)")
    print(f"• Total Inference Time: {inference_time:.3f}s ({inference_time/total_time*100:.1f}%)")
    print(
        f"• Vision Processing: {time_breakdown['Vision Model Prefill']:.3f}s ({time_breakdown['Vision Model Prefill']/total_time*100:.1f}%)"
    )

    if decode_tokens > 0:
        avg_decode_time = time_breakdown["Actual Inference Decode"] / decode_tokens
        print(f"• Average per-token decode: {avg_decode_time*1000:.2f}ms ({decode_tokens} tokens)")
        if time_breakdown["Other Decode Time"] > 0:
            total_decode_time = (
                time_breakdown["Compile Decode"]
                + time_breakdown["Actual Inference Decode"]
                + time_breakdown["Other Decode Time"]
            )
            print(
                f"• Decode overhead: {time_breakdown['Other Decode Time']:.3f}s ({time_breakdown['Other Decode Time']/total_decode_time*100:.1f}% of total decode time)"
            )

    # Performance bottlenecks
    max_component = max(time_breakdown.items(), key=lambda x: x[1])
    print(f"• Biggest bottleneck: {max_component[0]} ({max_component[1]/total_time*100:.1f}%)")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze Qwen2.5-VL performance metrics and create visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python perf_analysis.py data.json
  python perf_analysis.py metrics.json --output my_chart.png
  python perf_analysis.py data.json -o chart.png --no-show
        """,
    )

    parser.add_argument("json_file", help="JSON file containing performance metrics")
    parser.add_argument(
        "-o",
        "--output",
        default="qwen25_vl_performance_breakdown.png",
        help="Output PNG file name (default: qwen25_vl_performance_breakdown.png)",
    )
    parser.add_argument("--no-show", action="store_true", help="Don't display the chart window, just save to file")

    args = parser.parse_args()

    # Check if file exists
    json_file = args.json_file
    try:
        data = load_performance_data(json_file)
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        print("Usage: python perf_analysis.py <json_file>")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{json_file}': {e}")
        sys.exit(1)

    # Analyze time breakdown
    time_breakdown = analyze_time_breakdown(data)

    # Print analysis
    print_analysis_summary(time_breakdown, data)

    # Create pie chart
    fig = create_pie_chart(time_breakdown, data)

    # Save the plot
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"\nChart saved as '{args.output}'")

    # Show the plot unless --no-show is specified
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
