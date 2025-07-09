import pandas as pd
from collections import defaultdict
import sys

def analyze_zone_durations(csv_file, core_x, core_y, zone_name):
    """
    Analyze zone durations from profiled CSV data
    
    Parameters:
    csv_file (str): Path to the CSV file
    core_x (int): Core X coordinate
    core_y (int): Core Y coordinate  
    zone_name (str): Zone name to analyze
    
    Returns:
    dict: Dictionary containing min and max durations
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file, skiprows=1)

        df.columns = df.columns.str.strip()
        # Filter by core coordinates and zone name
        filtered_df = df[(df['core_x'] == core_x) & 
                        (df['core_y'] == core_y) & 
                        (df['zone name'] == zone_name)]
        
        if filtered_df.empty:
            print(f"No data found for core ({core_x}, {core_y}) and zone '{zone_name}'")
            return None
        
        # Separate ZONE_START and ZONE_END entries
        zone_starts = filtered_df[filtered_df['type'] == 'ZONE_START'].copy()
        zone_ends = filtered_df[filtered_df['type'] == 'ZONE_END'].copy()
        
        if zone_starts.empty or zone_ends.empty:
            print(f"Missing ZONE_START or ZONE_END entries for zone '{zone_name}'")
            return None
        
        # Sort by time to ensure proper matching
        zone_starts = zone_starts.sort_values('time[cycles since reset]')
        zone_ends = zone_ends.sort_values('time[cycles since reset]')
        
        # Calculate durations by matching START with END
        durations = []
        
        # Group by timer_id for accurate pairing
        start_groups = zone_starts.groupby('timer_id')
        end_groups = zone_ends.groupby('timer_id')
        
        for timer_id in start_groups.groups:
            if timer_id in end_groups.groups:
                start_times = start_groups.get_group(timer_id)['time[cycles since reset]'].values
                end_times = end_groups.get_group(timer_id)['time[cycles since reset]'].values
                
                # Match each start with corresponding end
                min_pairs = min(len(start_times), len(end_times))
                for i in range(min_pairs):
                    duration = end_times[i] - start_times[i]
                    if duration > 0:  # Only positive durations
                        durations.append(duration)
        print(durations)
        
        if not durations:
            print(f"No valid durations calculated for zone '{zone_name}'")
            return None
        
        # Calculate min and max durations
        min_duration = min(durations)
        max_duration = max(durations)
        
        # Convert to milliseconds (assuming 1000 MHz frequency)
        min_duration_ms = min_duration / 1000000
        max_duration_ms = max_duration / 1000000
        
        results = {
            'min_duration': min_duration,
            'max_duration': max_duration,
            'min_duration_ms': min_duration_ms,
            'max_duration_ms': max_duration_ms,
            'total_durations': len(durations),
            'all_durations': durations
        }
        
        return results
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def main():
    # Check command line arguments
    if len(sys.argv) != 5:
        print("Usage: python zone_duration_analyzer.py <csv_file> <core_x> <core_y> <zone_name>")
        print("Example: python zone_duration_analyzer.py sample_profile.csv 1 1 'WRITE_OUT'")
        return
    
    csv_file = sys.argv[1]
    core_x = int(sys.argv[2])
    core_y = int(sys.argv[3])
    zone_name = sys.argv[4]
    
    print(f"Analyzing zone '{zone_name}' for core ({core_x}, {core_y})")
    print("-" * 50)
    
    results = analyze_zone_durations(csv_file, core_x, core_y, zone_name)
    
    if results:
        print(f"Zone: {zone_name}")
        print(f"Total duration pairs: {results['total_durations']}")
        print(f"Min duration: {results['min_duration']:,} cycles ({results['min_duration_ms']:.3f} ms)")
        print(f"Max duration: {results['max_duration']:,} cycles ({results['max_duration_ms']:.3f} ms)")
        print(f"Average duration: {sum(results['all_durations'])/len(results['all_durations']):,.0f} cycles")
    else:
        print("No results found.")

if __name__ == "__main__":
    main()

# Simplified version for direct use
def simple_zone_analysis(csv_file, core_x, core_y, zone_name):
    """Simplified version that returns just min and max durations"""
    df = pd.read_csv(csv_file)
    filtered = df[(df['core_x'] == core_x) & (df['core_y'] == core_y) & (df['zone name'] == zone_name)]
    
    starts = filtered[filtered['type'] == 'ZONE_START']['time[cycles since reset]'].values
    ends = filtered[filtered['type'] == 'ZONE_END']['time[cycles since reset]'].values
    
    if len(starts) == 0 or len(ends) == 0:
        return None
    
    durations = [end - start for start, end in zip(starts, ends) if end > start]
    
    if durations:
        return {'min_duration': min(durations), 'max_duration': max(durations)}
    return None

