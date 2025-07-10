import sys
from krpsim import ProcessFileParser, load_and_parse_file
from collections import defaultdict
import os

def load_trace(trace_path: str):
    if not os.path.isfile(trace_path):
        print(f"Error: Trace file '{trace_path}' does not exist.")
        sys.exit(1)
    if not os.access(trace_path, os.R_OK):
        print(f"Error: Trace file '{trace_path}' is not readable (permission denied).")
        sys.exit(1)

    final_stocks = {}
    trace = []
    max_delay = None

    try:
        with open(trace_path, 'r', encoding='utf-8') as f:
            mode = None
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                if line.startswith("==="):
                    if "MAX DELAY" in line:
                        mode = "max"
                    elif "FINAL" in line:
                        mode = "stock"
                    elif "TRACE" in line:
                        mode = "trace"
                    continue

                if mode == "max":
                    try:
                        max_delay = int(line.strip())
                    except ValueError:
                        print(f"Error: Invalid MAX DELAY at line {line_num}: '{line}'")
                        sys.exit(1)
                elif mode == "stock":
                    if ':' not in line:
                        print(f"Error: Invalid stock line at line {line_num}: '{line}'")
                        sys.exit(1)
                    k, v = line.split(':', 1)
                    try:
                        final_stocks[k.strip()] = int(v.strip())
                    except ValueError:
                        print(f"Error: Invalid stock quantity at line {line_num}: '{line}'")
                        sys.exit(1)
                elif mode == "trace":
                    if not line.startswith("Time"):
                        print(f"Error: Invalid trace line at line {line_num}: '{line}'")
                        sys.exit(1)
                    parts = line.split(":")
                    try:
                        time = int(parts[0].replace("Time", "").strip())
                        name = ":".join(parts[1:]).strip()
                        trace.append((time, name))
                    except ValueError:
                        print(f"Error: Invalid time in trace at line {line_num}: '{line}'")
                        sys.exit(1)

        if max_delay is None:
            print(f"Error: Missing MAX DELAY in trace file '{trace_path}'.")
            sys.exit(1)
        if not final_stocks:
            print(f"Error: Missing FINAL STOCKS section in trace file '{trace_path}'.")
            sys.exit(1)
        if not trace:
            print(f"Error: Trace section is empty in trace file '{trace_path}'.")
            sys.exit(1)

    except Exception as e:
        print(f"Error: Failed to read trace file '{trace_path}': {e}")
        sys.exit(1)

    return max_delay, final_stocks, trace


def verify(parser: ProcessFileParser, final_stocks_ref: dict, trace: list, max_delay: int):
    processes = {p.name: p for p in parser.processes}
    stocks = parser.initial_stocks.copy()

    trace_by_time = defaultdict(list)
    for t, pname in trace:
        trace_by_time[t].append(pname)

    running_processes = []
    all_times = sorted(trace_by_time.keys())
    current_time_idx = 0

    while current_time_idx < len(all_times) or running_processes:
        if current_time_idx < len(all_times):
            t = all_times[current_time_idx]
        else:
            t = min(p[0] for p in running_processes)

        if t > max_delay:
            print(f"Stopped: reached MAX_DELAY {max_delay}")
            break

        finished_now = [p for p in running_processes if p[0] <= t]
        for end_time, pname in sorted(finished_now):
            if end_time > max_delay:
                continue
            p = processes[pname]
            for res, qty in p.outputs.items():
                stocks[res] = stocks.get(res, 0) + qty
            print(f"Finished at time {end_time}: {pname}")

        running_processes = [p for p in running_processes if p[0] > t]

        if t in trace_by_time:
            batch = trace_by_time[t]
            total_needs = defaultdict(int)

            for pname in batch:
                if pname == "no more process doable":
                    continue
                if pname not in processes:
                    print(f"ERROR: Process '{pname}' not found.")
                    return False
                p = processes[pname]
                for res, qty in p.inputs.items():
                    total_needs[res] += qty

            for res, total in total_needs.items():
                if stocks.get(res, 0) < total:
                    print(f"ERROR at time {t}: Not enough '{res}' (needed {total}, have {stocks.get(res, 0)})")
                    return False

            for pname in batch:
                if pname == "no more process doable":
                    continue
                p = processes[pname]
                for res, qty in p.inputs.items():
                    stocks[res] -= qty
                end_time = t + p.duration
                running_processes.append((end_time, pname))
                print(f"Started at time {t}: {pname} (will finish at {end_time})")

            current_time_idx += 1
        else:
            current_time_idx += 1

    for end_time, pname in sorted(running_processes):
        if end_time <= max_delay:
            print(f"UNFINISHED (should finish before max): {pname} at {end_time}")

    for res, qty in final_stocks_ref.items():
        if stocks.get(res, 0) != qty:
            print(f"ERROR: Final stock mismatch for '{res}': got {stocks.get(res, 0)}, expected {qty}")
            return False

    for res in stocks:
        if res not in final_stocks_ref and stocks[res] > 0:
            print(f"ERROR: Resource '{res}' present in final stocks but not expected (quantity: {stocks[res]}).")
            return False

    print("Trace verified successfully ✅")
    print(f"Final stocks: {stocks}")
    return True



def main():
    if len(sys.argv) != 3:
        print("Usage: krpsim_verif <file> <result_to_test>")
        sys.exit(1)

    config_path = sys.argv[1]
    trace_path = sys.argv[2]

    parser = load_and_parse_file(config_path)
    max_delay, final_stocks, trace = load_trace(trace_path)

    if not verify(parser, final_stocks, trace, max_delay):
        sys.exit(1)

if __name__ == "__main__":
    main()
