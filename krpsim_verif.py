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

    for t in sorted(trace_by_time.keys()):
        batch = trace_by_time[t]

        # Calculer besoins globaux
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

        # Vérifier faisabilité
        for res, total in total_needs.items():
            if stocks.get(res, 0) < total:
                print(f"ERROR at time {t}: Not enough '{res}' (needed {total}, have {stocks.get(res, 0)})")
                print(f"  Batch attempted: {[pname for pname in batch if pname != 'no more process doable']}")
                print(f"Final stocks: {stocks}")
                return False

        # Consommer inputs
        for pname in batch:
            if pname == "no more process doable":
                continue
            p = processes[pname]
            for res, qty in p.inputs.items():
                stocks[res] -= qty

        # Produire outputs et loguer
        for pname in batch:
            if pname == "no more process doable":
                continue
            p = processes[pname]
            for res, qty in p.outputs.items():
                stocks[res] = stocks.get(res, 0) + qty
            print(f"Executed at time {t}: {pname}")

    # Vérifier stocks finaux
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
