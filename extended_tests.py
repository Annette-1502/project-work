from pathlib import Path

from Problem import Problem
from tests import run_single_test, save_results



RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULT_FILE = RESULTS_DIR / "extended_test_results.csv"


#Questa suite estesa non sostituisce i test principali:
#serve solo a controllare qualche combinazione in piu
#su densita, seed e valori di beta vicini a 1
def main():
    tests = [
        #Grafo piu sparso con seed diverso dal principale
        Problem(200, density=0.1, alpha=1, beta=1, seed=7),
        Problem(200, density=0.1, alpha=2, beta=1, seed=7),

        #Casi intermedi per vedere come si comporta il solver
        #quando beta è vicino a 1 ma non esattamente uguale
        Problem(200, density=0.3, alpha=1, beta=1.1, seed=21),
        Problem(200, density=0.6, alpha=2, beta=1.3, seed=21),

        #Taglia media un po piu impegnativa senza arrivare ai 1000 nodi
        Problem(500, density=0.1, alpha=1, beta=1, seed=99),
        Problem(500, density=0.3, alpha=2, beta=1.1, seed=99),

        #Altri casi lineari con densita e seed diversi
        Problem(300, density=0.05, alpha=1, beta=1, seed=123),
        Problem(300, density=0.8, alpha=2, beta=1, seed=5),

        #Casi concavi: beta minore di 1 non ha un ramo dedicato nel solver
        #e viene usato per controllare la robustezza della strategia route-based
        Problem(200, density=0.3, alpha=1, beta=0.8, seed=31),
        Problem(500, density=0.2, alpha=2, beta=0.5, seed=44),

        #Valori intermedi di beta per controllare il passaggio verso il caso convesso
        Problem(300, density=0.2, alpha=1, beta=1.5, seed=33),
        Problem(700, density=0.2, alpha=2, beta=1.2, seed=17),

        #Controllo aggiuntivo su un caso convesso con seed diverso
        Problem(500, density=0.6, alpha=1, beta=2, seed=13),
    ]

    rows = []

    for index, problem in enumerate(tests, start=1):
        rows.append(run_single_test(index, problem))

    save_results(rows, RESULT_FILE)


if __name__ == "__main__":
    main()
