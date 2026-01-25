import subprocess
import csv
import io
import sys
import os
import argparse
from statistics import mean

def run_ncu_for_mode(mode, batch_size, in_features, out_features, n_bits, group_size):
    script_path = os.path.join(os.path.dirname(__file__), "profile_infra.py")
    
    metrics = [
        "gpu__time_duration.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    ]
    
    cmd = [
        "ncu",
        "--csv",
        "--profile-from-start", "off", # Important!
        "--metrics", ",".join(metrics),
        sys.executable, script_path,
        "--batch_size", str(batch_size),
        "--in_features", str(in_features),
        "--out_features", str(out_features),
        "--n_bits", str(n_bits),
        "--group_size", str(group_size),
        "--steps", "5",
        "--mode", mode
    ]
    
    print(f"Running profiling for [{mode}]...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error profiling {mode}:")
        print(e.stderr)
        return None

def parse_and_accumulate(csv_data, mode, all_kernels):
    if not csv_data: return

    f = io.StringIO(csv_data)
    reader = csv.reader(f)
    header = None
    
    rows = []
    for line in reader:
        if not line: continue
        if "Kernel Name" in line and "Metric Name" in line:
            header = line
            continue
        if header and len(line) == len(header):
            rows.append(dict(zip(header, line)))
    
    if not rows:
        print(f"No data for {mode}")
        return

    # Process rows
    for row in rows:
        k_name = row.get("Kernel Name")
        if not k_name: continue
        
        metric_name = row.get("Metric Name")
        metric_val_str = row.get("Metric Value")
        
        if not metric_name or not metric_val_str: continue
        try:
            val = float(metric_val_str.replace(',', ''))
        except ValueError:
            continue
            
        key = (mode, k_name)
        if key not in all_kernels:
            all_kernels[key] = {}
        
        if metric_name not in all_kernels[key]:
            all_kernels[key][metric_name] = []
        all_kernels[key][metric_name].append(val)

def print_summary(all_kernels):
    print("\n" + "="*95)
    print(f"{'Mode':<10} | {'Kernel Name':<50} | {'Dur (us)':<10} | {'MEM %':<6} | {'SM %':<6}")
    print("-" * 95)
    
    # Sort by mode then duration
    sorted_keys = sorted(all_kernels.keys(), key=lambda x: (x[0], -mean(all_kernels[x].get("gpu__time_duration.sum", [0]))))
    
    for mode, k_name in sorted_keys:
        metrics = all_kernels[(mode, k_name)]
        
        durations = metrics.get("gpu__time_duration.sum", [0])
        avg_dur_ns = mean(durations)
        avg_dur_us = avg_dur_ns / 1000.0
        
        mem_pcts = metrics.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", [0])
        avg_mem = mean(mem_pcts)
        
        sm_pcts = metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", [0])
        avg_sm = mean(sm_pcts)
        
        display_name = k_name
        if len(display_name) > 48:
            display_name = display_name[:45] + "..."
            
        print(f"{mode:<10} | {display_name:<50} | {avg_dur_us:<10.2f} | {avg_mem:<6.1f} | {avg_sm:<6.1f}")
    
    print("="*95)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--in_features", type=int, default=4096)
    parser.add_argument("--out_features", type=int, default=11008) 
    parser.add_argument("--n_bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    args = parser.parse_args()
    
    all_kernels = {}
    
    # Profile Forward
    csv_fwd = run_ncu_for_mode("forward", args.batch_size, args.in_features, args.out_features, args.n_bits, args.group_size)
    parse_and_accumulate(csv_fwd, "forward", all_kernels)
    
    # Profile Backward
    csv_bwd = run_ncu_for_mode("backward", args.batch_size, args.in_features, args.out_features, args.n_bits, args.group_size)
    parse_and_accumulate(csv_bwd, "backward", all_kernels)
    
    print_summary(all_kernels)