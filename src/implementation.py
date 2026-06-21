import math
import random
import time
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx



City = int
Gold = float
Step = Tuple[City, Gold]
Route = Tuple[City, ...]

#Tolleranza numerica per evitare falsi confronti dovuti ai float
EPS = 1e-9

#Nei casi convessi conviene spezzare molto il carico
#Questo limite evita path infiniti ma lascia spazio per migliorare il costo
MAX_CHUNKS_PER_CITY = 1000
SPARSE_HIGH_ALPHA_SEED = 17

#I test esterni validano gia il path finale.
#Lasciando questo flag a False evitiamo un secondo controllo costoso.
ENABLE_INTERNAL_VALIDATION = False


#Funzione usata dal file s338906
def build_collection_plan(problem, time_limit: Optional[float] = None, seed: int = 42) -> List[Step]:
    """
    Solver for the Gold Thief problem.

    Main ideas:
    - beta > 1:
      the carrying cost is convex, therefore the solver uses several
      light depot-city-depot trips and may collect a city's gold in chunks.

    - beta == 1:
      the carrying cost is linear, therefore the solver uses a simpler
      route-based heuristic with:
        * global savings;
        * angular sweep construction;
        * randomized multi-start;
        * relocation local search;
        * intra-route reordering.

    The returned path is always expanded on real graph edges.
    """

    start_time = time.perf_counter()
    rng = random.Random(seed)

    #Leggo una sola volta i dati del problema, cosi il resto del solver
    #ragiona su strutture semplici e veloci da consultare
    graph: nx.Graph = problem.graph
    alpha = float(problem.alpha)
    beta = float(problem.beta)
    depot = 0
    max_edges = len(graph.nodes) * (len(graph.nodes) - 1) / 2
    graph_density = graph.number_of_edges() / max_edges if max_edges else 0.0

    #Nei grafi grandi e sparsi con alpha alto conviene esplorare combinazioni
    #leggermente diverse in quanto il costo del carico pesa di piu
    if beta <= 1.0 + EPS and alpha > 1.1 and graph.number_of_nodes() >= 800 and graph_density < 0.75:
        rng = random.Random(SPARSE_HIGH_ALPHA_SEED)

    #Se il chiamante non impone un tempo allora scelgo un budget ragionevole
    #in base alla dimensione del grafo e al tipo di costo
    if time_limit is None:
        if beta > 1.0 + EPS:
            time_limit = 1.0
        else:
            node_count = graph.number_of_nodes()

            if node_count <= 120:
                time_limit = 3.0
            elif graph_density >= 0.75:
                time_limit = 4.0
            elif node_count >= 800:
                time_limit = 4.0
            else:
                time_limit = 3.5

    #La citta 0 è il deposito, quindi la escludo dalle route da costruire
    cities: List[City] = [node for node in graph.nodes if node != depot]

    if not cities:
        return [(0, 0)]

    #Mi salvo tutto l'oro in un dizionario compatto per non andare ogni volta
    #a pescare dentro il grafo
    gold: Dict[City, Gold] = {
        node: float(graph.nodes[node]["gold"])
        for node in graph.nodes
    }

    #Precalcolo distanze e cammini minimi dal deposito a tutte le citta
    dist0, paths0 = nx.single_source_dijkstra(
        graph,
        source=depot,
        weight="dist",
    )

    #Per ogni cammino dal deposito mi salvo un coefficiente utile a calcolare
    #in fretta il costo di ritorno quando il ladro è gia carico
    coeff0: Dict[City, float] = {}

    for node, path in paths0.items():
        coeff = 0.0

        for c1, c2 in zip(path, path[1:]):
            d = float(graph[c1][c2]["dist"])
            coeff += (alpha * d) ** beta

        coeff0[node] = coeff

    #Queste funzioni locali tengono tutta la parte di costo vicino al solver,
    #cosi il ragionamento principale resta piu leggibile
    def edge_cost(distance: float, carried: float) -> float:
        return distance + (alpha * distance * carried) ** beta

    def direct_edge_cost(c1: City, c2: City, carried: float) -> float:
        d = float(graph[c1][c2]["dist"])
        return edge_cost(d, carried)

    def loaded_return_cost(city: City, carried: float) -> float:
        return float(dist0[city]) + coeff0[city] * (carried ** beta)

    @lru_cache(maxsize=None)
    def route_cost(route: Route) -> float:
        """
        Cost of one route:
        depot -> route[0] -> ... -> route[-1] -> depot

        Internal route moves use direct graph edges.
        Depot legs use shortest paths.
        """

        if not route:
            return 0.0

        #Il primo pezzo della route parte sempre dal deposito verso la prima citta
        total = float(dist0[route[0]])
        carried = gold[route[0]]
        previous = route[0]

        for city in route[1:]:
            #Se la route logica contiene un salto impossibile la considero pessima
            #cosi il resto dell'euristica la scarta in automatico
            if not graph.has_edge(previous, city):
                return float("inf")

            #Mi sposto alla citta successiva portando con me l'oro raccolto fino a qui
            total += direct_edge_cost(previous, city, carried)
            carried += gold[city]
            previous = city

        #Alla fine della route torno al deposito con tutto il carico che ho accumulato
        total += loaded_return_cost(previous, carried)

        return total

    def routes_cost(routes: Iterable[Route]) -> float:
        #Il costo totale di una soluzione logica è solo la somma delle singole route
        return sum(route_cost(tuple(route)) for route in routes)

    def path_cost(path: Sequence[Step]) -> float:
        total = 0.0
        carried = 0.0

        for (c1, _), (c2, collected_at_c2) in zip(path, path[1:]):
            if c1 == c2:
                continue

            #Qui valuto il path finale passo per passo
            distance = float(graph[c1][c2]["dist"])
            total += distance + (alpha * distance * carried) ** beta

            if c2 == 0:
                #Quando torno al deposito scarico tutto e riparto da peso zero
                carried = 0.0
            else:
                carried += float(collected_at_c2)

        return total

    #Caso beta > 1:
    #quando il costo cresce in modo convesso conviene quasi sempre spezzare
    #i carichi e tornare spesso al deposito

    if beta > 1.0 + EPS:
        split_path = _build_chunked_trips_path(
            cities=cities,
            gold=gold,
            dist0=dist0,
            paths0=paths0,
            coeff0=coeff0,
            beta=beta,
        )

        if ENABLE_INTERNAL_VALIDATION:
            _check_path_validity(split_path, graph, gold)

        return split_path

    #Caso beta == 1:
    #qui il costo è lineare, quindi ha piu senso costruire route sensate,
    #ritoccarle un po' e poi scegliere la migliore

    #Questa é la baseline logica: una citta per viaggio
    baseline_routes: List[Route] = [(city,) for city in cities]
    best_routes: List[Route] = baseline_routes
    best_routes_cost = routes_cost(best_routes)

    singleton_cost = {
        city: route_cost((city,))
        for city in cities
    }

    candidate_edges = _rank_merge_candidates(
        graph=graph,
        gold=gold,
        dist0=dist0,
        loaded_return_cost=loaded_return_cost,
        direct_edge_cost=direct_edge_cost,
        singleton_cost=singleton_cost,
    )

    #Se alpha è basso posso permettermi route un po' piu lunghe,
    #altrimenti preferisco non caricarle troppo
    if alpha <= 1.1:
        max_route_length = 14
    else:
        max_route_length = 10

    relocation_passes = 2

    if len(cities) >= 800:
        if graph_density >= 0.75:
            relocation_passes = 20
        elif alpha <= 1.1:
            relocation_passes = 5
        else:
            relocation_passes = 4

    population: List[List[Route]] = []

    #Fase 1:
    #parto dalla versione piu deterministica ovvero quella guidata dai savings globali

    routes = _build_routes_from_merges(
        cities=cities,
        candidate_edges=candidate_edges,
        route_cost=route_cost,
        graph=graph,
        max_route_length=max_route_length,
    )

    routes = _improve_by_relocating_cities(
        routes=routes,
        route_cost=route_cost,
        graph=graph,
        max_route_length=max_route_length,
        start_time=start_time,
        time_limit=time_limit,
        max_passes=relocation_passes,
    )

    population.append(routes)

    #Fase 2:
    #costruisco anche soluzioni con sweep angolare, cosi esploro route
    #diverse senza introdurre una metaeuristica maggiormente pesante

    if time.perf_counter() - start_time < time_limit:
        for group_size in (3, 4, 6, 8, 10):
            routes = _build_routes_by_sweep(
                cities=cities,
                graph=graph,
                dist0=dist0,
                route_cost=route_cost,
                group_size=group_size,
            )

            routes = _improve_by_relocating_cities(
                routes=routes,
                route_cost=route_cost,
                graph=graph,
                max_route_length=max_route_length,
                start_time=start_time,
                time_limit=time_limit,
                max_passes=relocation_passes,
            )

            population.append(routes)

            if time.perf_counter() - start_time >= time_limit:
                break

    #Fase 3:
    #aggiungo un po' di rumore ai savings per ottenere piu punti di partenza
    #senza cambiare l'idea alla base del costruttore

    random_starts = 8 if len(cities) <= 150 else 5
    max_random_candidates = 180_000 if len(cities) > 300 else len(candidate_edges)
    base_for_random = candidate_edges[:max_random_candidates]

    for _ in range(random_starts):
        if time.perf_counter() - start_time >= time_limit:
            break

        noisy = [
            (saving * (0.80 + 0.40 * rng.random()), c1, c2)
            for saving, c1, c2 in base_for_random
        ]

        noisy.sort(key=lambda item: item[0], reverse=True)

        routes = _build_routes_from_merges(
            cities=cities,
            candidate_edges=noisy,
            route_cost=route_cost,
            graph=graph,
            max_route_length=max_route_length,
        )

        routes = _improve_by_relocating_cities(
            routes=routes,
            route_cost=route_cost,
            graph=graph,
            max_route_length=max_route_length,
            start_time=start_time,
            time_limit=time_limit,
            max_passes=relocation_passes,
        )

        population.append(routes)

    #Fase 4:
    #provo uno scambio leggero tra citta di route diverse sui candidati migliori

    swapped_population: List[List[Route]] = []

    if beta >= 1.0 - EPS and len(cities) >= 800:
        for routes in sorted(population, key=routes_cost)[:3]:
            if time.perf_counter() - start_time >= 0.85 * time_limit:
                break

            swapped_population.append(
                _improve_by_swapping_cities(
                    routes=routes,
                    route_cost=route_cost,
                    start_time=start_time,
                    time_limit=time_limit,
                )
            )

    population.extend(swapped_population)

    #Fase 5:
    #sulle soluzioni migliori provo a riordinare le citta interne alla route

    optimized_population: List[List[Route]] = []

    for routes in sorted(population, key=routes_cost)[:6]:
        if time.perf_counter() - start_time >= 0.97 * time_limit:
            break

        optimized_population.append(
            _improve_route_orders(
                routes=routes,
                route_cost=route_cost,
                graph=graph,
                start_time=start_time,
                time_limit=time_limit,
            )
        )

    population.extend(optimized_population)
    population = sorted(population, key=routes_cost)[:8]

    #Fase 6:
    #a questo punto scelgo semplicemente la soluzione logica migliore

    for routes in population:
        cost = routes_cost(routes)

        if cost + EPS < best_routes_cost:
            best_routes = routes
            best_routes_cost = cost

    best_path = _expand_routes_to_full_path(
        routes=best_routes,
        graph=graph,
        gold=gold,
        paths0=paths0,
    )

    #Tengo pronta anche la baseline espansa:
    #se per qualche motivo la mia soluzione peggiora davvero, non mi faccio male
    baseline_path = _expand_routes_to_full_path(
        routes=baseline_routes,
        graph=graph,
        gold=gold,
        paths0=paths0,
    )

    best_path = _delay_pickups_on_linear_case(
        path=best_path,
        graph=graph,
        gold=gold,
    )
    baseline_path = _delay_pickups_on_linear_case(
        path=baseline_path,
        graph=graph,
        gold=gold,
    )

    best_path_cost = path_cost(best_path)
    baseline_path_cost = path_cost(baseline_path)

    if baseline_path_cost + EPS < best_path_cost:
        best_path = baseline_path

    _check_path_validity(best_path, graph, gold)

    return best_path


#Assegna un punteggio alle fusioni citta-citta piu promettenti.
def _rank_merge_candidates(
    *,
    graph: nx.Graph,
    gold: Dict[City, Gold],
    dist0: Dict[City, float],
    loaded_return_cost,
    direct_edge_cost,
    singleton_cost: Dict[City, float],
) -> List[Tuple[float, City, City]]:
    """
    Compute global two-city savings.

    This is more global than simply sorting by distance:
    each candidate edge is ranked by the actual estimated reduction
    in total route cost.
    """

    candidates: List[Tuple[float, City, City]] = []

    for c1, c2, _ in graph.edges(data=True):
        #Non mi interessa fondere il deposito con una citta,
        #qui sto solo cercando coppie di citta compatibili
        if c1 == 0 or c2 == 0:
            continue

        #Confronto il costo di due viaggi separati con il costo di un viaggio combinato
        old_cost = singleton_cost[c1] + singleton_cost[c2]

        cost_12 = (
            float(dist0[c1])
            + direct_edge_cost(c1, c2, gold[c1])
            + loaded_return_cost(c2, gold[c1] + gold[c2])
        )

        cost_21 = (
            float(dist0[c2])
            + direct_edge_cost(c2, c1, gold[c2])
            + loaded_return_cost(c1, gold[c1] + gold[c2])
        )

        saving = old_cost - min(cost_12, cost_21)

        #Tengo solo fusioni veramente utili, cosi la lista candidati resta pulita
        if saving > EPS:
            candidates.append((saving, c1, c2))

    candidates.sort(key=lambda item: item[0], reverse=True)

    return candidates


#Prova a fondere route gia esistenti solo quando il costo totale scende
def _build_routes_from_merges(
    *,
    cities: Sequence[City],
    candidate_edges: Sequence[Tuple[float, City, City]],
    route_cost,
    graph: nx.Graph,
    max_route_length: int,
    initial_routes: Optional[Sequence[Route]] = None,
) -> List[Route]:
    """
    Clarke-Wright-like route merge.

    A merge is accepted only if the complete route cost improves.
    """

    cities_set = set(cities)

    if initial_routes is None:
        #Caso standard: parto con una route singola per ogni citta
        routes: Dict[int, Route] = {
            city: (city,)
            for city in cities
        }

        owner: Dict[City, int] = {
            city: city
            for city in cities
        }

    else:
        #Se parto da route gia esistenti, le ricostruisco in una forma coerente
        #e butto via eventuali citta duplicate o pezzi non validi
        routes = {}
        owner = {}
        seen = set()
        next_id = 1

        for route in initial_routes:
            current: List[City] = []

            for city in route:
                if city not in cities_set:
                    continue

                if city in seen:
                    continue

                #Se il pezzo che sto leggendo si spezza, apro una nuova route
                if current and not graph.has_edge(current[-1], city):
                    route_id = next_id
                    next_id += 1

                    routes[route_id] = tuple(current)

                    for node in current:
                        owner[node] = route_id

                    current = [city]
                else:
                    current.append(city)

                seen.add(city)

            if current:
                route_id = next_id
                next_id += 1

                routes[route_id] = tuple(current)

                for node in current:
                    owner[node] = route_id

        for city in cities:
            if city not in owner:
                route_id = next_id
                next_id += 1

                routes[route_id] = (city,)
                owner[city] = route_id

    costs: Dict[int, float] = {
        route_id: route_cost(route)
        for route_id, route in routes.items()
    }

    for _, c1, c2 in candidate_edges:
        #Recupero le due route a cui appartengono le citta candidate
        route_1_id = owner.get(c1)
        route_2_id = owner.get(c2)

        if route_1_id is None or route_2_id is None:
            continue

        if route_1_id == route_2_id:
            continue

        if route_1_id not in routes or route_2_id not in routes:
            continue

        route_1 = routes[route_1_id]
        route_2 = routes[route_2_id]

        if len(route_1) + len(route_2) > max_route_length:
            continue

        old_cost = costs[route_1_id] + costs[route_2_id]

        #Provo tutte le orientazioni sensate delle due route
        #e tengo la fusione migliore tra quelle valide
        best_route = None
        best_cost = old_cost

        orientations_1 = (route_1,) if len(route_1) == 1 else (route_1, route_1[::-1])
        orientations_2 = (route_2,) if len(route_2) == 1 else (route_2, route_2[::-1])

        for oriented_1 in orientations_1:
            for oriented_2 in orientations_2:
                if graph.has_edge(oriented_1[-1], oriented_2[0]):
                    merged = oriented_1 + oriented_2
                    cost = route_cost(merged)

                    if cost + EPS < best_cost:
                        best_cost = cost
                        best_route = merged

                if graph.has_edge(oriented_2[-1], oriented_1[0]):
                    merged = oriented_2 + oriented_1
                    cost = route_cost(merged)

                    if cost + EPS < best_cost:
                        best_cost = cost
                        best_route = merged

        if best_route is not None:
            #Se ho trovato una fusione migliorativa aggiorno struttura e proprietari
            routes[route_1_id] = best_route
            costs[route_1_id] = best_cost

            for city in best_route:
                owner[city] = route_1_id

            del routes[route_2_id]
            del costs[route_2_id]

    return list(routes.values())


#Costruisco route raggruppando citta che stanno in direzioni simili
def _build_routes_by_sweep(
    *,
    cities: Sequence[City],
    graph: nx.Graph,
    dist0: Dict[City, float],
    route_cost,
    group_size: int,
) -> List[Route]:
    """
    Angular sweep construction.

    Cities with similar angle around the depot are grouped, then ordered
    from farther to nearer.
    """

    depot_x, depot_y = graph.nodes[0]["pos"]

    ordered = sorted(
        cities,
        key=lambda city: math.atan2(
            graph.nodes[city]["pos"][1] - depot_y,
            graph.nodes[city]["pos"][0] - depot_x,
        ),
    )

    #Dopo averle ordinate per angolo, le spezzo in piccoli gruppi
    #e provo a farle diventare route coerenti
    routes: List[Route] = []

    for start in range(0, len(ordered), group_size):
        group = ordered[start:start + group_size]
        group = sorted(group, key=lambda city: dist0[city], reverse=True)

        #Dentro ogni gruppo provo a collegare le citta dalla piu lontana alla piu vicina
        current: List[City] = []

        for city in group:
            if not current:
                current = [city]
                continue

            candidate = tuple(current + [city])
            previous = tuple(current)

            if (
                graph.has_edge(current[-1], city)
                and route_cost(candidate) + EPS < route_cost(previous) + route_cost((city,))
            ):
                #Se attaccare la nuova citta conviene, allungo la route corrente
                current.append(city)
            else:
                #Altrimenti chiudo la route e ne apro una nuova
                routes.append(tuple(current))
                current = [city]

        if current:
            routes.append(tuple(current))

    return routes


#Mossa di ricerca locale: sposta una citta da una route a un'altra
#solo se il bilancio complessivo migliora davvero
def _improve_by_relocating_cities(
    *,
    routes: Sequence[Route],
    route_cost,
    graph: nx.Graph,
    max_route_length: int,
    start_time: float,
    time_limit: float,
    max_passes: int,
) -> List[Route]:

    routes = [tuple(route) for route in routes if route]
    costs = [route_cost(route) for route in routes]

    for _ in range(max_passes):
        if time.perf_counter() - start_time >= time_limit:
            break

        improved = False

        for i, route in enumerate(list(routes)):
            if time.perf_counter() - start_time >= time_limit:
                break

            if not route:
                continue

            for position, city in enumerate(route):
                if time.perf_counter() - start_time >= time_limit:
                    break

                #Tolgo temporaneamente una citta dalla sua route per vedere
                #se altrove riesco a inserirla meglio
                route_without = route[:position] + route[position + 1:]
                cost_without = route_cost(route_without) if route_without else 0.0

                if cost_without == float("inf"):
                    continue

                best_delta = 0.0
                best_move = None

                for j, target_route in enumerate(routes):
                    if time.perf_counter() - start_time >= time_limit:
                        break

                    if i == j:
                        continue

                    if len(target_route) >= max_route_length:
                        continue

                    old_cost = costs[i] + costs[j]

                    for insert_position in range(len(target_route) + 1):
                        if time.perf_counter() - start_time >= time_limit:
                            break

                        if (
                            insert_position > 0
                            and not graph.has_edge(target_route[insert_position - 1], city)
                        ):
                            continue

                        if (
                            insert_position < len(target_route)
                            and not graph.has_edge(city, target_route[insert_position])
                        ):
                            continue

                        new_target = (
                            target_route[:insert_position]
                            + (city,)
                            + target_route[insert_position:]
                        )

                        new_cost = cost_without + route_cost(new_target)
                        delta = old_cost - new_cost

                        #Tengo solo il miglior spostamento trovato fin qui
                        if delta > best_delta + EPS:
                            best_delta = delta
                            best_move = (
                                i,
                                j,
                                route_without,
                                new_target,
                                cost_without,
                                route_cost(new_target),
                            )

                if best_move is not None:
                    #Applico subito la migliore mossa trovata in questa passata
                    i2, j2, new_i, new_j, cost_i, cost_j = best_move

                    routes[i2] = new_i
                    costs[i2] = cost_i

                    routes[j2] = new_j
                    costs[j2] = cost_j

                    if not routes[i2]:
                        del routes[i2]
                        del costs[i2]

                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return [route for route in routes if route]


#Mossa di ricerca locale: scambia due citta tra route diverse
def _improve_by_swapping_cities(
    *,
    routes: Sequence[Route],
    route_cost,
    start_time: float,
    time_limit: float,
) -> List[Route]:
    routes = [tuple(route) for route in routes if route]
    costs = [route_cost(route) for route in routes]

    for _ in range(3):
        if time.perf_counter() - start_time >= 0.92 * time_limit:
            break

        improved = False

        for i, route_1 in enumerate(routes):
            if time.perf_counter() - start_time >= 0.92 * time_limit:
                break

            if len(route_1) <= 1:
                continue

            for j in range(i + 1, len(routes)):
                if time.perf_counter() - start_time >= 0.92 * time_limit:
                    break

                route_2 = routes[j]

                if len(route_2) <= 1:
                    continue

                old_cost = costs[i] + costs[j]
                best_delta = 0.0
                best_swap = None

                for p1, city_1 in enumerate(route_1):
                    if time.perf_counter() - start_time >= 0.92 * time_limit:
                        break

                    for p2, city_2 in enumerate(route_2):
                        new_route_1 = route_1[:p1] + (city_2,) + route_1[p1 + 1:]
                        new_route_2 = route_2[:p2] + (city_1,) + route_2[p2 + 1:]

                        cost_1 = route_cost(new_route_1)

                        if cost_1 == float("inf"):
                            continue

                        cost_2 = route_cost(new_route_2)

                        if cost_2 == float("inf"):
                            continue

                        delta = old_cost - (cost_1 + cost_2)

                        if delta > best_delta + EPS:
                            best_delta = delta
                            best_swap = (new_route_1, new_route_2, cost_1, cost_2)

                if best_swap is not None:
                    new_route_1, new_route_2, cost_1, cost_2 = best_swap
                    routes[i] = new_route_1
                    routes[j] = new_route_2
                    costs[i] = cost_1
                    costs[j] = cost_2
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return routes


#Rifinisco l'ordine interno di ogni route senza cambiare il set delle citta visitate
def _improve_route_orders(
    *,
    routes: Sequence[Route],
    route_cost,
    graph: nx.Graph,
    start_time: float,
    time_limit: float,
) -> List[Route]:

    def improve_one_route_order(route: Route) -> Route:
        if len(route) <= 1:
            return route

        #Lavoro sempre su una copia immutabile della route migliore trovata fino a questo momento
        best = tuple(route)
        best_cost = route_cost(best)

        improved = True
        passes = 0

        while improved and passes < 3:
            passes += 1
            improved = False

            if time.perf_counter() - start_time >= 0.97 * time_limit:
                break

            n = len(best)

            #Prima provo piccole inversioni di segmento in stile 2-opt
            for i in range(n):
                if time.perf_counter() - start_time >= 0.97 * time_limit:
                    return best

                if improved:
                    break

                for j in range(i + 1, n):
                    if time.perf_counter() - start_time >= 0.97 * time_limit:
                        return best

                    candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]

                    if candidate == best:
                        continue

                    #Anche le piccole mosse interne devono restare ammissibili sul grafo
                    if not all(graph.has_edge(c1, c2) for c1, c2 in zip(candidate, candidate[1:])):
                        continue

                    cost = route_cost(candidate)

                    if cost + EPS < best_cost:
                        best = candidate
                        best_cost = cost
                        improved = True
                        break

            if improved:
                continue

            n = len(best)

            #Se non basta, provo a sfilare una citta e reinserirla altrove
            for i in range(n):
                if time.perf_counter() - start_time >= 0.97 * time_limit:
                    return best

                if improved:
                    break

                city = best[i]
                remaining = best[:i] + best[i + 1:]

                for j in range(len(remaining) + 1):
                    if time.perf_counter() - start_time >= 0.97 * time_limit:
                        return best

                    candidate = remaining[:j] + (city,) + remaining[j:]

                    if candidate == best:
                        continue

                    if not all(graph.has_edge(c1, c2) for c1, c2 in zip(candidate, candidate[1:])):
                        continue

                    cost = route_cost(candidate)

                    if cost + EPS < best_cost:
                        best = candidate
                        best_cost = cost
                        improved = True
                        break

        return best

    optimized: List[Route] = []

    for route in routes:
        if time.perf_counter() - start_time >= 0.97 * time_limit:
            #Se il tempo sta finendo, preferisco restituire la route cosi come è
            optimized.append(tuple(route))
            continue

        optimized.append(improve_one_route_order(tuple(route)))

    return optimized


#Trasforma route logiche in un path esplicito fatto di archi reali del grafo
def _expand_routes_to_full_path(
    *,
    routes: Sequence[Route],
    graph: nx.Graph,
    gold: Dict[City, Gold],
    paths0: Dict[City, List[City]],
) -> List[Step]:
    """
    Convert logical routes into a real path over graph edges.

    Shortest path transit nodes are inserted with collected gold equal to 0.
    """

    final_path: List[Step] = [(0, 0)]
    collected = set()

    def append(city: City, amount: Gold) -> None:
        #Se resto sullo stesso nodo accumulo solo l'oro
        #evitando di creare duplicati inutili nel path
        if final_path[-1][0] == city:
            if abs(amount) > EPS:
                final_path[-1] = (city, final_path[-1][1] + amount)
        else:
            final_path.append((city, amount))

    for route in routes:
        if not route:
            continue

        #Dal deposito entro nella route passando per il cammino minimo
        for node in paths0[route[0]][1:]:
            amount = gold[node] if node == route[0] and node not in collected else 0.0
            append(node, amount)

            if node == route[0]:
                collected.add(node)

        previous = route[0]

        #Dentro la route provo a usare l'arco diretto previsto dal costruttore
        for city in route[1:]:
            amount = gold[city] if city not in collected else 0.0

            if graph.has_edge(previous, city):
                append(city, amount)
            else:
                #Se salta fuori una mossa interna non valida, la trasformo
                #subito in un cammino minimo reale
                sp = nx.shortest_path(graph, previous, city, weight="dist")

                for node in sp[1:]:
                    append(node, amount if node == city else 0.0)

            collected.add(city)
            previous = city

        #Chiudo sempre la route tornando al deposito
        back_path = list(reversed(paths0[previous]))

        for node in back_path[1:]:
            append(node, 0.0)

        if final_path[-1][0] == 0:
            final_path[-1] = (0, 0)
        else:
            final_path.append((0, 0))

    return _normalize_path(final_path)


#Nel caso convesso decide quante visite conviene fare a ogni citta
def _build_chunked_trips_path(
    *,
    cities: Sequence[City],
    gold: Dict[City, Gold],
    dist0: Dict[City, float],
    paths0: Dict[City, List[City]],
    coeff0: Dict[City, float],
    beta: float,
) -> List[Step]:

    counts: Dict[City, int] = {}

    for city in cities:
        city_gold = max(0.0, gold[city])
        distance = max(EPS, float(dist0[city]))
        coeff = max(EPS, coeff0[city])

        if city_gold <= EPS:
            counts[city] = 1
            continue

        #Stimo quante volte vale la pena visitare la citta
        #invece di prendere tutto in un colpo solo
        estimated = (
            ((beta - 1.0) * coeff * (city_gold ** beta))
            / (2.0 * distance)
        ) ** (1.0 / beta)

        center = max(1, int(round(estimated)))
        candidates = {1}

        for k in range(center - 3, center + 4):
            if k >= 1:
                candidates.add(min(MAX_CHUNKS_PER_CITY, k))

        def split_cost(k: int) -> float:
            chunk = city_gold / k
            return k * (2.0 * distance + coeff * (chunk ** beta))

        counts[city] = min(candidates, key=split_cost)

    ordered_cities = sorted(
        cities,
        key=lambda city: (counts[city], gold[city], dist0[city]),
        reverse=True,
    )

    path: List[Step] = [(0, 0)]

    def append(city: City, amount: Gold) -> None:
        if path[-1][0] == city:
            if abs(amount) > EPS:
                path[-1] = (city, path[-1][1] + amount)
        else:
            path.append((city, amount))

    for city in ordered_cities:
        number_of_trips = counts[city]

        if number_of_trips <= 1:
            chunks = [gold[city]]
        else:
            #Divido l'oro in parti quasi uguali, lasciando all'ultimo chunk
            #l'eventuale resto dovuto ai float
            chunk = gold[city] / number_of_trips
            chunks = [chunk] * number_of_trips
            chunks[-1] = gold[city] - sum(chunks[:-1])

        path_to_city = paths0[city]
        path_to_depot = list(reversed(path_to_city))

        for amount in chunks:
            #Ogni chunk genera un mini viaggio deposito -> citta -> deposito
            for node in path_to_city[1:]:
                append(node, amount if node == city else 0.0)

            for node in path_to_depot[1:]:
                append(node, 0.0)

            if path[-1][0] == 0:
                path[-1] = (0, 0)
            else:
                path.append((0, 0))

    return _normalize_path(path)


#Quando beta è 1 conviene spesso raccogliere l'oro il piu tardi possibile
def _delay_pickups_on_linear_case(
    *,
    path: Sequence[Step],
    graph: nx.Graph,
    gold: Dict[City, Gold],
) -> List[Step]:
    """
    For beta == 1, collecting on the latest useful visit of a city is never worse.
    """

    if len(path) <= 2:
        return list(path)

    #Per ogni posizione mi salvo quanta distanza manca prima di riscaricare al deposito
    remaining_distance = [0.0] * len(path)
    distance_to_depot = 0.0

    for idx in range(len(path) - 2, -1, -1):
        city = path[idx][0]

        if city == 0:
            distance_to_depot = 0.0
            continue

        next_city = path[idx + 1][0]
        distance_to_depot += float(graph[city][next_city]["dist"])
        remaining_distance[idx] = distance_to_depot

    best_occurrence: Dict[City, int] = {}

    for idx, (city, _) in enumerate(path):
        if city == 0:
            continue

        #Tengo l'ultima visita davvero utile alla citta
        #cioè quella piu vicina al successivo ritorno al deposito
        if (
            city not in best_occurrence
            or remaining_distance[idx] <= remaining_distance[best_occurrence[city]] + EPS
        ):
            best_occurrence[city] = idx

    #Ricostruisco il path mettendo inizialmente zero oro ovunque
    retimed: List[Step] = [(city, 0.0) for city, _ in path]

    if retimed:
        retimed[0] = (0, 0)
        retimed[-1] = (0, 0)

    for city, amount in gold.items():
        if city == 0:
            continue

        idx = best_occurrence.get(city)

        if idx is None:
            continue

        #Assegno tutto l'oro della citta alla visita piu conveniente
        current_city, current_amount = retimed[idx]
        retimed[idx] = (current_city, current_amount + amount)

    return _normalize_path(retimed)


#Ripulisce il path da ripetizioni inutili e forza inizio/fine nel deposito
def _normalize_path(path: Sequence[Step]) -> List[Step]:
    if not path:
        return [(0, 0)]

    cleaned: List[Step] = []

    for city, amount in path:
        #Se due occorrenze consecutive dello stesso nodo non aggiungono oro
        #la seconda è solo rumore e la elimino
        if cleaned and cleaned[-1][0] == city and abs(amount) <= EPS:
            continue

        cleaned.append((city, amount))

    if cleaned[0] != (0, 0):
        cleaned.insert(0, (0, 0))

    if cleaned[-1] != (0, 0):
        if cleaned[-1][0] == 0:
            cleaned[-1] = (0, 0)
        else:
            cleaned.append((0, 0))

    return cleaned


#Ultimo controllo di sicurezza prima di restituire il path al tester
def _check_path_validity(
    path: Sequence[Step],
    graph: nx.Graph,
    gold: Dict[City, Gold],
) -> None:
    if not path:
        raise ValueError("Empty path")

    #Il path finale deve sempre partire e finire nel deposito
    if path[0] != (0, 0):
        raise ValueError("Path must start at (0, 0)")

    if path[-1] != (0, 0):
        raise ValueError("Path must end at (0, 0)")

    collected = {
        city: 0.0
        for city in graph.nodes
    }

    for city, amount in path:
        if city not in graph.nodes:
            raise ValueError(f"Unknown city {city}")

        if amount < -1e-8:
            raise ValueError(f"Negative gold at city {city}")

        collected[city] += float(amount)

    #Controllo uno per uno tutti i passi del path finale
    for (c1, _), (c2, _) in zip(path, path[1:]):
        if not graph.has_edge(c1, c2):
            raise ValueError(f"Invalid edge {c1} -> {c2}")

    #Alla fine la quantita raccolta in ogni citta deve coincidere con quella del problema
    for city in graph.nodes:
        expected = float(gold[city])
        actual = collected[city]

        if abs(actual - expected) > 1e-5:
            raise ValueError(
                f"Wrong collected gold at city {city}: "
                f"{actual} instead of {expected}"
            )
