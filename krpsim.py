import random   
import copy
from typing import List, Dict, Tuple
import re
import gc
import sys
import os

class Process:
    def __init__(self, name: str, inputs: Dict[str, int], outputs: Dict[str, int], duration: int):
        self.name = name
        self.inputs = inputs  
        self.outputs = outputs  
        self.duration = duration
        
    def can_execute(self, stocks: Dict[str, int]) -> bool:
        """Vérifie si le processus peut être exécuté avec les stocks actuels"""
        for resource, needed in self.inputs.items():
            if stocks.get(resource, 0) < needed:
                return False
        return True
    
    def execute(self, stocks: Dict[str, int]) -> Dict[str, int]:
        """Exécute le processus et retourne les nouveaux stocks"""
        new_stocks = stocks.copy()
        
        # Consommer les ressources d'entrée
        for resource, needed in self.inputs.items():
            new_stocks[resource] -= needed
            
        # Ajouter les ressources de sortie
        for resource, produced in self.outputs.items():
            new_stocks[resource] = new_stocks.get(resource, 0) + produced
            
        return new_stocks

class ProcessFileParser:
    def __init__(self):
        self.initial_stocks = {}
        self.processes = []
        self.optimize_targets = []


    def find_child(self, index, to_find, target):
        for process in self.processes:
            max_output_key = max(process.outputs.items(), key=lambda item: item[1])[0]
            if max_output_key in to_find and max_output_key != target:
                if process not in self.pathes[index]:
                    self.pathes[index].append(process)
                    self.find_child(index, process.inputs, target)

    def get_all_resource_types(self) -> List[str]:
        all_resources = set(self.initial_stocks.keys())
        
        for process in self.processes:
            all_resources.update(process.inputs.keys())
            all_resources.update(process.outputs.keys())
        
        return list(all_resources)
    
    def parse_file(self, content: str):
        lines = content.strip().split('\n')

        if not lines:
            raise ValueError("The file is empty after stripping whitespace.")

        has_stock = False
        has_process = False
        has_optimize = False

        for idx, line in enumerate(lines, start=1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if ':' in line and not line.startswith('optimize:'):
                if '(' not in line:
                    parts = line.split(':')
                    if len(parts) != 2:
                        raise ValueError(f"Invalid stock declaration on line {idx}: '{line}'")
                    resource, quantity = parts
                    resource = resource.strip()
                    quantity = quantity.strip()
                    if not resource:
                        raise ValueError(f"Empty stock name on line {idx}.")
                    if not quantity.isdigit():
                        raise ValueError(f"Invalid stock quantity on line {idx}: '{quantity}'")
                    self.initial_stocks[resource] = int(quantity)
                    has_stock = True
                else:
                    self.parse_process(line)
                    has_process = True
            elif line.startswith('optimize:'):
                targets_str = line.replace('optimize:', '').strip()
                if not targets_str:
                    raise ValueError(f"No optimization target specified on line {idx}.")
                if not (targets_str.startswith('(') and targets_str.endswith(')')):
                    raise ValueError(f"Optimization line must use parentheses on line {idx}.")
                targets_str = targets_str[1:-1]
                targets = targets_str.split(';')
                for target in targets:
                    target = target.strip()
                    if target:
                        self.optimize_targets.append(target)
                has_optimize = True

        if not has_stock:
            raise ValueError("No initial stocks defined in the file.")
        if not has_process:
            raise ValueError("No process definitions found in the file.")
        if not has_optimize:
            raise ValueError("No optimization targets found in the file.")

        # Pick target
        for targ in self.optimize_targets:
            if "time" not in targ:
                self.target = targ

    def parse_process(self, line: str):
        line = line.strip()
        pattern = r'^(\w+):\((.*?)\):\((.*?)\):(\d+)$'
        match = re.match(pattern, line)

        if not match:
            raise ValueError(f"Invalid process format: '{line}'")

        name = match.group(1).strip()
        inputs_str = match.group(2).strip()
        outputs_str = match.group(3).strip()
        duration_str = match.group(4).strip()

        if not name:
            raise ValueError(f"Process name is empty in line: '{line}'")

        try:
            duration = int(duration_str)
            if duration <= 0:
                raise ValueError
        except ValueError:
            raise ValueError(f"Invalid duration in process: '{duration_str}' in line: '{line}'")

        inputs = self.parse_resources(inputs_str, context=f"inputs of process '{name}'")
        outputs = self.parse_resources(outputs_str, context=f"outputs of process '{name}'")

        process = Process(name, inputs, outputs, duration)
        self.processes.append(process)

    def parse_resources(self, resource_str: str, context: str = "") -> Dict[str, int]:
        resources = {}
        if resource_str.strip():
            for item in resource_str.split(';'):
                item = item.strip()
                if not item:
                    continue

                if ':' not in item:
                    raise ValueError(f"Invalid resource entry '{item}' in {context}.")

                resource, quantity = item.split(':', 1)
                resource = resource.strip()
                quantity = quantity.strip()

                if not resource:
                    raise ValueError(f"Empty resource name in {context}.")
                if not quantity.isdigit() or int(quantity) <= 0:
                    raise ValueError(f"Invalid quantity '{quantity}' for resource '{resource}' in {context}.")

                if resource in resources:
                    raise ValueError(f"Duplicate resource '{resource}' in {context}.")

                resources[resource] = int(quantity)

        return resources

class Individual:
    def __init__(self, process_counts: Dict[str, int], origin):
        self.process_counts = process_counts  # Nombre de fois qu'on exécute chaque processus
        self.fitness = 0
        self.total_time = 0
        self.final_stocks = {}
        self.executed_counts = {}
        self.process_counts_origin = origin

    def __str__(self):
        return f"Counts: {self.process_counts}, Fitness: {self.fitness}, Time: {self.total_time}"

class GeneticAlgorithm:
    def __init__(self, parser: ProcessFileParser, population_size=50, generations=450, 
                 mutation_rate=0.2, crossover_rate=0.8, time_limit=50000):
        self.parser = parser
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.time_limit = time_limit
        self.population = []
        self.first_fifties = 1
        self.best_fits = "unknown"
        self.old_fitness = 0
        self.stuck = 0

        # Analyser la rentabilité des processus
        self.analyze_profitability()
        self.analyze_profitability_optimize_target()

    def analyze_profitability(self):
        """Analyse la rentabilité de chaque processus"""
        self.process_profitability = {}

        initial_resources = set(self.parser.initial_stocks.keys())
        
        for process in self.parser.processes:
            profit = 0
            cost = 0
            
            # Calculer le coût en target
            cost += process.inputs.get(self.parser.target, 0)
            
            bonus = 0
            for resource in process.inputs:
                if resource in initial_resources:
                    amount_used = process.inputs[resource]
                    bonus += amount_used
            profit += bonus * 2

            profit += process.outputs.get(self.parser.target, 0)
            
            # Rentabilité = gain / (coût + durée)
            if cost > 0 or process.duration > 0:
                self.process_profitability[process.name] = profit / (cost + process.duration / 100)
            else:
                self.process_profitability[process.name] = profit
        
        print("Analyse de rentabilité:")
        for name, profit in sorted(self.process_profitability.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {profit:.3f}")


    def analyze_profitability_optimize_target(self):
        self.target_profitability = {}
        initial_resources = set(self.parser.initial_stocks.keys())
        for process in self.parser.processes:
            profit = (process.outputs.get(self.parser.target, 0) * 10) + 1
            cost = process.inputs.get(self.parser.target, 0)
            
            bonus = 0
            for resource in process.inputs:
                if resource in initial_resources:
                    amount_used = process.inputs[resource]
                    bonus += amount_used
            profit -= bonus / 100

            if cost > 0 or process.duration > 0:
                self.target_profitability[process.name] = profit / (cost + process.duration / 100)
            else:
                self.target_profitability[process.name] = profit

        print("\nAnalyse de rentabilité avec priorité à la target:")
        for name, profit in sorted(self.target_profitability.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {profit:.3f}")
        
    def create_random_individual(self, use_target=False) -> Individual:
        """Crée un individu basé soit sur la rentabilité classique, soit sur la cible uniquement"""
        process_counts = {}
        
        profitability = self.target_profitability if use_target else self.process_profitability
        profitable_processes = [name for name, profit in profitability.items() if profit > 0]
        
        for process in self.parser.processes:
            if process.name in profitable_processes:
                max_count = random.randint(100, 1000) if profitability[process.name] > 1 else random.randint(1, 100)
            else:
                max_count = random.randint(1, 100)
            process_counts[process.name] = random.randint(0, max_count)
        
        if use_target:
            return Individual(process_counts, origin="target")
        else:
            return Individual(process_counts, origin="general")
    
    def simulate_execution(self, individual: Individual):
        """Simule l'exécution avec gestion du parallélisme"""
        individual.executed_counts = {}
        stocks = self.parser.initial_stocks.copy()
        current_time = 0
        running_processes = []  # [(end_time, process, outputs)]
        remaining_counts = individual.process_counts.copy()
        process_map = {p.name: p for p in self.parser.processes}
        
        while current_time < self.time_limit:
            # Terminer les processus finis
            finished = [p for p in running_processes if p[0] <= current_time]
            for end_time, process, outputs in finished:
                for resource, quantity in outputs.items():
                    stocks[resource] = stocks.get(resource, 0) + quantity
            
            running_processes = [p for p in running_processes if p[0] > current_time]
            
            # Lancer de nouveaux processus
            launched_any = True
            while launched_any and current_time < self.time_limit:
                launched_any = False
                
                # Trier les processus par priorité (rentabilité)
                process_priority = [(name, count) for name, count in remaining_counts.items() 
                                  if count > 0 and name in process_map]
                if individual.process_counts_origin == 'general':
                    process_priority.sort(key=lambda x: self.process_profitability.get(x[0], 0), reverse=True)
                else :
                    process_priority.sort(key=lambda x: self.target_profitability.get(x[0], 0), reverse=True)
                
                unique_priority = []
                for name, count in process_priority:
                    if name not in unique_priority:
                        unique_priority.append(name)

                for process_name, count in process_priority:
                    if count <= 0:
                        continue
                    
                    if len(unique_priority) > 0:
                        next_process = unique_priority.pop(0)
                        process = process_map[next_process]
                    else:
                        process = process_map[process_name]
                    
                    if process.can_execute(stocks):
                        # Consommer les inputs
                        for resource, needed in process.inputs.items():
                            stocks[resource] -= needed
                        
                        # Programmer la fin du processus
                        end_time = current_time + process.duration
                        running_processes.append((end_time, process, process.outputs))
                        
                        individual.executed_counts[process.name] = individual.executed_counts.get(process.name, 0) + 1

                        remaining_counts[process_name] -= 1
                        launched_any = True
                        
                        if end_time > self.time_limit:
                            break
            
                # Avancer le temps
            if running_processes:
                current_time = min(p[0] for p in running_processes)
            else:
                # Construire les stocks futurs (actuels + outputs des processus en cours)
                future_stocks = stocks.copy()
                for end_time, process, outputs in running_processes:
                    for res, qty in outputs.items():
                        future_stocks[res] = future_stocks.get(res, 0) + qty

                # Y a-t-il un processus exécutable avec ces stocks futurs ?
                possible_future = False
                for process_name, count in remaining_counts.items():
                    if count <= 0:
                        continue
                    process = process_map[process_name]
                    if all(future_stocks.get(res, 0) >= qty for res, qty in process.inputs.items()):
                        possible_future = True
                        break

                if possible_future:
                    # On avance le temps d'une unité (ou tu pourrais chercher la prochaine opportunité possible)
                    current_time += 1
                else:
                    break


        
        individual.total_time = min(current_time, self.time_limit)
        individual.final_stocks = stocks

        process_map = {p.name: p for p in self.parser.processes}
    
    def evaluate_fitness(self, individual: Individual):
        """Évalue la fitness d'un individu"""
        self.simulate_execution(individual)
        
        fitness = 0
        stocks = individual.final_stocks
        
        target = ''
        for targ in self.parser.optimize_targets:
            if targ != 'time':
                target = targ

        # Fitness basée sur les objectifs
        target_value = stocks.get(target, 0)
        fitness += target_value
        if 'time' in self.parser.optimize_targets:
                # Pénaliser le temps long
            fitness -= individual.total_time / 1000
        
        individual.fitness = fitness

    
    def tournament_selection(self, tournament_size=3) -> Individual:
        """Sélection par tournoi"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Croisement uniforme"""
        if random.random() > self.crossover_rate:
            child1 = Individual(process_counts=dict(parent1.process_counts), origin = parent1.process_counts_origin)
            child2 = Individual(process_counts=dict(parent2.process_counts), origin = parent1.process_counts_origin)
            return child1, child2
        
        child1_counts = {}
        child2_counts = {}
        
        all_processes = set(parent1.process_counts.keys()) | set(parent2.process_counts.keys())
        
        for process in all_processes:
            count1 = parent1.process_counts.get(process, 0)
            count2 = parent2.process_counts.get(process, 0)
            
            if random.random() < 0.5:
                child1_counts[process] = count1
                child2_counts[process] = count2
            else:
                child1_counts[process] = count2
                child2_counts[process] = count1
        
        return Individual(child1_counts, origin = parent1.process_counts_origin), Individual(child2_counts, origin = parent2.process_counts_origin)
    
    def mutate(self, individual: Individual):
        """Mutation des compteurs"""
        for process_name in individual.process_counts:
            if random.random() < self.mutation_rate:
                # Mutation forte pour les processus rentables
                if individual.process_counts_origin == 'general':
                    if self.process_profitability.get(process_name, 0) > 1:
                        individual.process_counts[process_name] += random.randint(-50, 200)
                    else:
                        individual.process_counts[process_name] += random.randint(-20, 50)
                else:
                    if self.target_profitability.get(process_name, 0) > 1:
                        individual.process_counts[process_name] += random.randint(-50, 200)
                    else:
                        individual.process_counts[process_name] += random.randint(-20, 50)
                # Garder les valeurs positives
                individual.process_counts[process_name] = max(0, individual.process_counts[process_name])

    def count_origins(self):
        target_based = 0
        general_based = 0
        for ind in self.population:
            origin = ind.process_counts_origin if hasattr(ind, "process_counts_origin") else "unknown"
            if origin == "target":
                target_based += 1
            else:
                general_based += 1
        if target_based == general_based:
            self.best_fits = 'unknown'
        if target_based > general_based:
            self.best_fits = 'target'
        else:
            self.best_fits = 'general'
        print(f"Origine des individus: target={target_based}, général={general_based}")

    def run(self):
        """Exécute l'algorithme génétique"""
        print("Création de la population initiale...")
        self.population = []

        if self.best_fits == "unknown":
            half = self.population_size // 2
            for _ in range(half):
                self.population.append(self.create_random_individual(use_target=False))
            for _ in range(self.population_size - half):
                self.population.append(self.create_random_individual(use_target=True))
        elif self.best_fits == 'target':
            self.population = [self.create_random_individual(use_target=True) for _ in range(self.population_size)]
        else:
            self.population = [self.create_random_individual(use_target=False) for _ in range(self.population_size)]
        
        print("Évaluation initiale...")
        for individual in self.population:
            self.evaluate_fitness(individual)
        
        
        for generation in range(self.generations):
            # Nouvelle population
            new_population = []
            
            # Élitisme
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            elite_size = max(1, self.population_size // 10)
            elite = self.population[:elite_size]
            new_population.extend([Individual(dict(ind.process_counts), origin=ind.process_counts_origin) for ind in elite])

            
            # Générer le reste
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                child1, child2 = self.crossover(parent1, parent2)
                
                self.mutate(child1)
                self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            new_population = new_population[:self.population_size]
            
            # Évaluer
            for individual in new_population:
                self.evaluate_fitness(individual)
            
            self.population = new_population
            
            # Statistiques
            best_individual = max(self.population, key=lambda x: x.fitness)
            
            del new_population
            del elite_size
            if generation % 50 == 0:
                print(self.best_fits)
                if self.first_fifties:
                    self.first_fifties = 0
                    self.count_origins()
                target = best_individual.final_stocks.get(self.parser.target, 0)
                if int(best_individual.fitness) <= int(self.old_fitness):
                    self.stuck += 1
                    if self.stuck == 2:
                        return best_individual
                else:
                    self.stuck = 0
                    self.old_fitness = best_individual.fitness
                print(f"Génération {generation}: Fitness = {best_individual.fitness:.0f}, {self.parser.target} = {target}, Temps = {best_individual.total_time}")
                gc.collect()

        best_individual = max(self.population, key=lambda x: x.fitness)
        return best_individual


def load_file_content(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"The file '{path}' does not exist or is not a valid file.")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"The file '{path}' cannot be read (permission denied).")
    if os.path.getsize(path) == 0:
        raise ValueError(f"The file '{path}' is empty.")
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
        if not content.strip():
            raise ValueError(f"The file '{path}' is empty or contains only whitespace.")
        return content

def main():
    if len(sys.argv) != 3:
        print("Usage: krpsim <file> <delay>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        time_limit = int(sys.argv[2])
        if time_limit <= 0:
            raise ValueError
    except ValueError:
        print("Error: The delay must be a positive integer.")
        sys.exit(1)

    try:
        file_content = load_file_content(file_path)
    except (FileNotFoundError, PermissionError, ValueError) as e:
        print(f"Error while opening the file: {e}")
        sys.exit(1)

    parser = ProcessFileParser()

    try:
        parser.parse_file(file_content)
    except Exception as e:
        print(f"Parsing error: {e}")
        sys.exit(1)

    print(f"File loaded: {len(parser.processes)} processes, {len(parser.get_all_resource_types())} resources")
    print(f"Optimization targets: {parser.optimize_targets}")

    print("\n=== RUNNING GENETIC ALGORITHM ===")
    ga = GeneticAlgorithm(parser, population_size=20, generations=450,
                          mutation_rate=0.2, time_limit=time_limit)
    best_solution = ga.run()

    print("\n" + "="*60)
    print("BEST SOLUTION FOUND:")
    print("="*60)
    print(f"Fitness: {best_solution.fitness:.0f}")
    print(f"Time used: {best_solution.total_time}/{ga.time_limit}")
    print(f"Final {parser.target}: {best_solution.final_stocks.get(parser.target, 0)}")

    print("\nProcesses executed:")
    for process_name, count in sorted(best_solution.executed_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  {process_name}: {count}")

    print("\nFinal stocks:")
    for resource, quantity in best_solution.final_stocks.items():
        if quantity > 0:
            print(f"  {resource}: {quantity}")
    
if __name__ == "__main__":
    main()