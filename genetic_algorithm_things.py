from random import choices, randint, randrange, random
from typing import List, Optional, Callable, Tuple

Genome = List[int] #UN GENOME SE REFIERE A UN MODELO (SET DE HYPERAPARAMETROS)
Population = List[Genome] #VARIOS MODELOS (CADA UNO DISTINTO HYPERPARAMETROS)
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[Genome], int] 
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
PrinterFunc = Callable[[Population, int, FitnessFunc], None]


# LENGTH  ES LA CANTIDAD DE ELEMENTOS 
def generate_genome(length: int) -> Genome:
    return choices([0, 1], k=length)


def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]


def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of same length")

    length = len(a)
    if length < 2:
        return a, b

    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]


def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome


def population_fitness(population: Population, fitness_func: FitnessFunc) -> int:
    return sum([fitness_func(genome) for genome in population])


def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(gene) for gene in population],
        k=2
    )


def sort_population(population: Population, fitness_func: FitnessFunc) -> Population:
    return sorted(population, key=fitness_func, reverse=True)


def genome_to_string(genome: Genome) -> str:
    return "".join(map(str, genome))


def print_stats(population: Population, generation_id: int, fitness_func: FitnessFunc):
    print("GENERATION %02d" % generation_id)
    print("=============")
    print("Population: [%s]" % ", ".join([genome_to_string(gene) for gene in population]))
    print("Avg. Fitness: %f" % (population_fitness(population, fitness_func) / len(population)))
    sorted_population = sort_population(population, fitness_func)
    print(
        "Best: %s (%f)" % (genome_to_string(sorted_population[0]), fitness_func(sorted_population[0])))
    print("Worst: %s (%f)" % (genome_to_string(sorted_population[-1]),
                              fitness_func(sorted_population[-1])))
    print("")

    return sorted_population[0]


def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: int,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100,
        printer: Optional[PrinterFunc] = None) \
        -> Tuple[Population, int]:
    population = populate_func()

    for i in range(generation_limit):
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)
        
        if printer is not None:
            printer(population, i, fitness_func)
        
        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[0:2]
        
        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation
        
    return population, i


#======things======
from collections import namedtuple
Thing = namedtuple('Thing', ['name', 'value', 'weight'])

def generate_things(num: int) -> [Thing]:
    return [Thing(f"thing{i}", i, i) for i in range(1, num+1)]



def fitness(genome: Genome, things: [Thing], weight_limit: int) -> int:
    
    if len(genome) != len(things):
        raise ValueError("genome and things must be of same length")

    weight = 0
    value = 0
    for i, thing in enumerate(things):
        if genome[i] == 1:
            weight += thing.weight
            value += thing.value
            
            if weight > weight_limit:
                
                return 0
    
    return value
    
def from_genome(genome: Genome, things: [Thing]) -> [Thing]:
    result = []
    for i, thing in enumerate(things):
        if genome[i] == 1:
            result += [thing]

    return result

def print_stats(things: [Thing]):
    print(f"Things: {to_string(things)}")
    print("Values: ",{t.value for t in things})
    print("Weights: ",{t.weight for t in things})
    

def to_string(things: [Thing]):
    return f"[{', '.join([t.name for t in things])}]"

#======end things======
from functools import partial

things=generate_things(10)

population, generations = run_evolution(
    populate_func=partial(
        generate_population,size=10,genome_length=len(things)
    ),
    fitness_func=partial(
        fitness,things=things,weight_limit=4
    ),
    fitness_limit=1000,
    generation_limit=1000
)

sack=from_genome(population[0], things)
print_stats(sack)

