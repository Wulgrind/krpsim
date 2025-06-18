import random
import copy
from typing import List, Dict, Tuple
import re

class Process:
    def __init__(self, name: str, inputs: Dict[str, int], outputs: Dict[str, int], duration: int):
        self.name = name
        self.inputs = inputs  # ressources nécessaires
        self.outputs = outputs  # ressources produites
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

    def get_all_resource_types(self) -> List[str]:
        all_resources = set(self.initial_stocks.keys())
        
        for process in self.processes:
            all_resources.update(process.inputs.keys())
            all_resources.update(process.outputs.keys())
        
        return list(all_resources)
    
    def parse_file(self, content: str):
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if ':' in line and not line.startswith('optimize:'):
                # Parse stock initial ou processus
                if '(' not in line:
                    # Stock initial
                    parts = line.split(':')
                    if len(parts) == 2:
                        resource, quantity = parts
                        self.initial_stocks[resource.strip()] = int(quantity.strip())
                else:
                    # Processus
                    self.parse_process(line)
            elif line.startswith('optimize:'):
                # Parse cibles d'optimisation - format: optimize:(target1;target2;...)
                targets_str = line.replace('optimize:', '').strip()
                if targets_str.startswith('(') and targets_str.endswith(')'):
                    targets_str = targets_str[1:-1]  # Enlever les parenthèses
                targets = targets_str.split(';')
                for target in targets:
                    target = target.strip()
                    if target:
                        self.optimize_targets.append(target)
    
    def parse_process(self, line: str):
        # Format: nom:(inputs):(outputs):duration
        # Exemple: process1:(iron:2;wood:1):(sword:1):5
        
        # Utiliser regex pour parser la ligne
        pattern = r'(\w+):\(([^)]*)\):\(([^)]*)\):(\d+)'
        match = re.match(pattern, line)
        
        if match:
            name = match.group(1)
            inputs_str = match.group(2)
            outputs_str = match.group(3)
            duration = int(match.group(4))
            
            inputs = self.parse_resources(inputs_str)
            outputs = self.parse_resources(outputs_str)
            
            process = Process(name, inputs, outputs, duration)
            self.processes.append(process)
    
    def parse_resources(self, resource_str: str) -> Dict[str, int]:
        resources = {}
        if resource_str.strip():
            for item in resource_str.split(';'):
                if ':' in item:
                    resource, quantity = item.split(':')
                    resources[resource.strip()] = int(quantity.strip())
        return resources

class Individual:
    def __init__(self, process_sequence: List[str]):
        self.process_sequence = process_sequence
        self.fitness = 0
        self.total_time = 0
        self.final_stocks = {}
    
    def __str__(self):
        return f"Sequence: {self.process_sequence}, Fitness: {self.fitness}, Time: {self.total_time}"

class GeneticAlgorithm:
    def __init__(self, parser: ProcessFileParser, population_size=100, generations=1000, 
                 mutation_rate=0.1, crossover_rate=0.8):
        self.parser = parser
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        
    def create_random_individual(self) -> Individual:
        stocks = self.parser.initial_stocks.copy()
        sequence = []
        attempts = 0

        while attempts < 100:  # éviter boucles infinies
            valid_processes = [p for p in self.parser.processes if p.can_execute(stocks)]
            if not valid_processes:
                break
            process = random.choice(valid_processes)
            sequence.append(process.name)
            stocks = process.execute(stocks)
            attempts += 1
        
        return Individual(sequence)
    
    def evaluate_fitness(self, individual: Individual):
        stocks = self.parser.initial_stocks.copy()
        total_time = 0
        executed_processes = []

        for process_name in individual.process_sequence:
            process = next((p for p in self.parser.processes if p.name == process_name), None)
            if process and process.can_execute(stocks):
                stocks = process.execute(stocks)
                total_time += process.duration
                executed_processes.append(process_name)

        individual.total_time = total_time
        individual.final_stocks = stocks

        # Calcul de la fitness
        fitness = 0

        for target in self.parser.optimize_targets:
            if target == 'time':
                if executed_processes:
                    fitness += max(0, 1000 - total_time)
            else:
                fitness += stocks.get(target, 0) * 100  # mettre un poids fort pour motiver

        # Bonus selon le nombre de processus exécutés
        fitness += len(executed_processes) * 10

        # Forte pénalité si aucun processus valide
        if not executed_processes:
            fitness -= 500
        
        individual.fitness = fitness

    
    def tournament_selection(self, tournament_size=3) -> Individual:
        """Sélection par tournoi"""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Croisement à un point"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        # Point de croisement
        min_len = min(len(parent1.process_sequence), len(parent2.process_sequence))
        if min_len < 2:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
            
        crossover_point = random.randint(1, min_len - 1)
        
        # Créer les enfants
        child1_sequence = (parent1.process_sequence[:crossover_point] + 
                          parent2.process_sequence[crossover_point:])
        child2_sequence = (parent2.process_sequence[:crossover_point] + 
                          parent1.process_sequence[crossover_point:])
        
        return Individual(child1_sequence), Individual(child2_sequence)
    
    def mutate(self, individual: Individual):
        """Mutation par remplacement aléatoire"""
        if random.random() < self.mutation_rate:
            if individual.process_sequence:
                # Remplacer un processus aléatoire
                index = random.randint(0, len(individual.process_sequence) - 1)
                process_names = [p.name for p in self.parser.processes]
                individual.process_sequence[index] = random.choice(process_names)
                
        # Mutation d'ajout/suppression
        if random.random() < self.mutation_rate / 2:
            if len(individual.process_sequence) > 1 and random.random() < 0.5:
                # Supprimer un processus
                individual.process_sequence.pop(random.randint(0, len(individual.process_sequence) - 1))
            else:
                # Ajouter un processus
                process_names = [p.name for p in self.parser.processes]
                individual.process_sequence.append(random.choice(process_names))
    
    def run(self):
        """Exécute l'algorithme génétique"""
        # Initialiser la population
        self.population = [self.create_random_individual() for _ in range(self.population_size)]
        
        # Évaluer la population initiale
        for individual in self.population:
            self.evaluate_fitness(individual)
        
        best_fitness_history = []
        
        for generation in range(self.generations):
            # Nouvelle population
            new_population = []
            
            # Élitisme - garder les meilleurs
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            elite_size = self.population_size // 10
            new_population.extend(copy.deepcopy(self.population[:elite_size]))
            
            # Générer le reste de la population
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                child1, child2 = self.crossover(parent1, parent2)
                
                self.mutate(child1)
                self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Limiter la taille de la population
            new_population = new_population[:self.population_size]
            
            # Évaluer la nouvelle population
            for individual in new_population:
                self.evaluate_fitness(individual)
            
            self.population = new_population
            
            # Statistiques
            best_individual = max(self.population, key=lambda x: x.fitness)
            best_fitness_history.append(best_individual.fitness)
            
            if generation % 100 == 0:
                print(f"Génération {generation}: Meilleure fitness = {best_individual.fitness}")
                print(f"Meilleure séquence: {best_individual.process_sequence[:10]}...")
                print(f"Temps total: {best_individual.total_time}")
                print(f"Stocks finaux: {best_individual.final_stocks}")
                print("-" * 50)
        
        # Retourner le meilleur individu
        best_individual = max(self.population, key=lambda x: x.fitness)
        return best_individual, best_fitness_history


def load_file_content(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

# Exemple d'utilisation
def main():

    file_path = "resources/simple"  
    file_content = load_file_content(file_path)

    parser = ProcessFileParser()
    parser.parse_file(file_content)


    print(f"Nice file! {len(parser.processes)} processes, {len(parser.get_all_resource_types())} stocks, {round(len(parser.optimize_targets) / 2)} to optimize")

    # Exécuter l'algorithme génétique
    print("=== EXÉCUTION DE L'ALGORITHME GÉNÉTIQUE ===")
    ga = GeneticAlgorithm(parser, population_size=100, generations=300, mutation_rate=0.15)
    best_solution, fitness_history = ga.run()

    print("\n" + "="*50)
    print("MEILLEURE SOLUTION TROUVÉE:")
    print("="*50)
    print(f"Séquence optimale: {best_solution.process_sequence}")
    print(f"Fitness: {best_solution.fitness}")
    print(f"Temps total: {best_solution.total_time}")
    print(f"Stocks finaux: {best_solution.final_stocks}")

    # Analyser la solution en détail
    print("\n=== ANALYSE DÉTAILLÉE DE LA SOLUTION ===")
    stocks = parser.initial_stocks.copy()
    total_time = 0

    print(f"État initial: {stocks}")

    for i, process_name in enumerate(best_solution.process_sequence):
        process = next((p for p in parser.processes if p.name == process_name), None)
        if process and process.can_execute(stocks):
            old_stocks = stocks.copy()
            stocks = process.execute(stocks)
            total_time += process.duration
            print(f"Étape {i+1}: {process_name}")
            print(f"  Avant: {old_stocks} -> Après: {stocks}")
            print(f"  Temps: +{process.duration} (total: {total_time})")
        else:
            print(f"Étape {i+1}: {process_name} - IMPOSSIBLE (ressources insuffisantes)")

    print(f"\nRésultat final: {len([p for p in best_solution.process_sequence if next((pr for pr in parser.processes if pr.name == p), None) and next((pr for pr in parser.processes if pr.name == p), None).can_execute(parser.initial_stocks)])} processus exécutables")
    for target in parser.optimize_targets:
        if target != "time":
            print(f"{target.capitalize()} produits: {stocks.get(target, 0)}")

    print(f"Temps total: {total_time}")

if __name__ == "__main__":
    main()