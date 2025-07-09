import sys
from krpsim import ProcessFileParser, load_and_parse_file
from collections import defaultdict

def load_trace(trace_path: str):
    final_stocks = {}
    trace = []

    with open(trace_path, 'r', encoding='utf-8') as f:
        mode = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("==="):
                if "FINAL" in line:
                    mode = "stock"
                elif "TRACE" in line:
                    mode = "trace"
                continue

            if mode == "stock":
                if ':' in line:
                    k, v = line.split(':', 1)
                    final_stocks[k.strip()] = int(v.strip())
            elif mode == "trace":
                if line.startswith("Time"):
                    parts = line.split(":")
                    time = int(parts[0].replace("Time", "").strip())
                    name = ":".join(parts[1:]).strip()
                    trace.append((time, name))

    return final_stocks, trace


def verify(parser: ProcessFileParser, final_stocks_ref: dict, trace: list):
    processes = {p.name: p for p in parser.processes}
    stocks = parser.initial_stocks.copy()

    trace_by_time = defaultdict(list)
    for t, pname in trace:
        trace_by_time[t].append(pname)

    running_processes = []  # liste de tuples (end_time, pname)

    all_times = sorted(trace_by_time.keys())
    current_time_idx = 0

    while current_time_idx < len(all_times) or running_processes:
        # Si on a encore des events, on avance au prochain temps de trace
        if current_time_idx < len(all_times):
            t = all_times[current_time_idx]
        else:
            # Plus de trace → avancer au prochain end_time
            t = min(p[0] for p in running_processes)

        # Terminer tous les process prévus jusqu’à maintenant
        finished_now = [p for p in running_processes if p[0] <= t]
        for end_time, pname in sorted(finished_now):
            p = processes[pname]
            for res, qty in p.outputs.items():
                stocks[res] = stocks.get(res, 0) + qty
            print(f"Finished at time {end_time}: {pname}")
        running_processes = [p for p in running_processes if p[0] > t]

        # Si on a une trace pour ce temps, traiter le batch
        if t in trace_by_time:
            batch = trace_by_time[t]

            total_needs = defaultdict(int)
            for pname in batch:
                if pname == "no more process doable":
                    continue

                if pname not in processes:
                    print(f"ERROR: Process '{pname}' not found in config.")
                    print(f"Final stocks: {stocks}")
                    return False

                p = processes[pname]
                for res, qty in p.inputs.items():
                    total_needs[res] += qty

            for res, total in total_needs.items():
                if stocks.get(res, 0) < total:
                    print(f"ERROR at time {t}: Not enough '{res}' (needed {total}, have {stocks.get(res, 0)})")
                    print(f"  Batch attempted: {[pname for pname in batch if pname != 'no more process doable']}")
                    print(f"Final stocks: {stocks}")
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
            # Si on est ici car on avait juste des process à finir → avancer
            current_time_idx += 1

    # Finir tout ce qui reste (en théorie tout est déjà terminé)
    if running_processes:
        for end_time, pname in sorted(running_processes):
            print(f"UNFINISHED: {pname} (should finish at {end_time})")
            p = processes[pname]
            for res, qty in p.outputs.items():
                stocks[res] = stocks.get(res, 0) + qty

    # Vérifier stock final
    for res, qty in final_stocks_ref.items():
        if stocks.get(res, 0) != qty:
            print(f"ERROR: Final stock mismatch for '{res}': got {stocks.get(res, 0)}, expected {qty}")
            print(f"Final stocks: {stocks}")
            return False

    print("Trace verified successfully. ✅")
    print(f"Final stocks: {stocks}")
    return True



def main():
    if len(sys.argv) != 3:
        print("Usage: krpsim_verif <config_file> <trace_file>")
        sys.exit(1)

    config_path = sys.argv[1]
    trace_path = sys.argv[2]

    parser = load_and_parse_file(config_path)

    final_stocks, trace = load_trace(trace_path)

    if not verify(parser, final_stocks, trace):
        sys.exit(1)

if __name__ == "__main__":
    main()
