
import csv
import importlib
from pathlib import Path
import time

from Problem import Problem


STUDENT_MODULE = "s338906"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULT_FILE = RESULTS_DIR / "test_results.csv"


solution = importlib.import_module(STUDENT_MODULE).solution



#ogni passo del path finale deve corrispondere a un arco reale del grafo
def is_valid(problem: Problem, path):
    graph = problem.graph

    for (c1, _), (c2, _) in zip(path, path[1:]):
        yield graph.has_edge(c1, c2)


#Questa funzione controlla che la soluzione sia corretta prima ancora di valutarla
def validate(problem: Problem, path):
    """
    Validate that the returned path respects the project rules:
    - starts at (0, 0);
    - ends at (0, 0);
    - uses only existing graph edges;
    - collects exactly all the gold in every city.
    """

    graph = problem.graph

    #Controllo subito la forma generale del risultato
    assert isinstance(path, list), "The solution must return a list"
    assert len(path) >= 1, "The path is empty"

    assert path[0] == (0, 0), "The path does not start from (0, 0)"
    assert path[-1] == (0, 0), "The path does not end in (0, 0)"

    #Ogni elemento deve essere proprio una coppia (citta, oro raccolto)
    for item in path:
        assert isinstance(item, tuple), f"Path item is not a tuple: {item}"
        assert len(item) == 2, f"Path item does not have length 2: {item}"

        city, amount = item

        assert city in graph.nodes, f"Unknown city in path: {city}"
        assert amount >= -1e-9, f"Negative gold collected in city {city}: {amount}"

    #Qui verifico la cosa piu importante per l'ammissibilita:
    #due passi consecutivi del path devono essere collegati da un arco vero
    if not all(is_valid(problem, path)):
        for (c1, _), (c2, _) in zip(path, path[1:]):
            assert graph.has_edge(c1, c2), f"Invalid edge: {c1} -> {c2}"

    #Alla fine ricontrollo che tutto l'oro del problema sia stato raccolto
    collected = {city: 0.0 for city in graph.nodes}

    for city, amount in path:
        collected[city] += amount

    for city in graph.nodes:
        expected = graph.nodes[city]["gold"]
        actual = collected[city]

        assert abs(expected - actual) < 1e-5, (
            f"Wrong gold in city {city}: collected {actual}, expected {expected}"
        )

    return True


#Questa funzione ricalcola il costo del path con la stessa formula del problema
def evaluate(problem: Problem, path):

    graph = problem.graph
    alpha = problem.alpha
    beta = problem.beta

    #Il carico cresce lungo il cammino e si azzera solo quando torno al deposito
    carried = 0.0
    total_cost = 0.0

    for (c1, _), (c2, g2) in zip(path, path[1:]):
        distance = graph[c1][c2]["dist"]
        total_cost += distance + (alpha * distance * carried) ** beta

        if c2 == 0:
            carried = 0.0
        else:
            carried += g2

    return total_cost


#Qui eseguo un singolo esperimento e stampo un riepilogo leggibile
def run_single_test(index, problem):
    node_count = len(problem.graph.nodes)

    print(f"Test {index}")
    print(
        f"Problem("
        f"n={node_count}, "
        f"alpha={problem.alpha}, "
        f"beta={problem.beta}"
        f")"
    )

    start = time.time()
    path = solution(problem)
    elapsed = time.time() - start

    #Prima valido il path, poi confronto il suo costo con la baseline
    validate(problem, path)

    baseline = problem.baseline()
    score = evaluate(problem, path)

    improvement = (baseline - score) / baseline * 100

    print(f"Baseline:    {baseline:.4f}")
    print(f"Solution:    {score:.4f}")
    print(f"Improvement: {improvement:.4f}%")
    print(f"Path length: {len(path)}")
    print(f"Time:        {elapsed:.4f} s")
    print("-" * 60)

    return {
        "test": index,
        "n": node_count,
        "alpha": problem.alpha,
        "beta": problem.beta,
        "baseline": f"{baseline:.4f}",
        "solution": f"{score:.4f}",
        "improvement_percent": f"{improvement:.4f}",
        "path_length": len(path),
        "time_seconds": f"{elapsed:.4f}",
    }


#Scrivo i risultati in CSV
def save_results(rows, output_file):
    RESULTS_DIR.mkdir(exist_ok=True)

    fieldnames = [
        "test",
        "n",
        "alpha",
        "beta",
        "baseline",
        "solution",
        "improvement_percent",
        "path_length",
        "time_seconds",
    ]

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


#Piccolo benchmark che usiamo per confrontare diverse versioni di problema
def main():
    tests = [
        Problem(100, density=0.2, alpha=1, beta=1),
        Problem(100, density=0.2, alpha=2, beta=1),
        Problem(100, density=0.2, alpha=1, beta=2),
        Problem(100, density=1, alpha=1, beta=1),
        Problem(100, density=1, alpha=2, beta=1),
        Problem(100, density=1, alpha=1, beta=2),
        Problem(1_000, density=0.2, alpha=1, beta=1),
        Problem(1_000, density=0.2, alpha=2, beta=1),
        Problem(1_000, density=0.2, alpha=1, beta=2),
        Problem(1_000, density=1, alpha=1, beta=1),
        Problem(1_000, density=1, alpha=2, beta=1),
        Problem(1_000, density=1, alpha=1, beta=2),
    ]

    rows = []

    for index, problem in enumerate(tests, start=1):
        rows.append(run_single_test(index, problem))

    save_results(rows, RESULT_FILE)


if __name__ == "__main__":
    main()
