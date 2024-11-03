# Поиск в ширину
from collections import deque

def bfs(graph, start, goal):
    queue = deque([start])
    visited = {start}
    parent = {start: None}

    while queue:
        vertex = queue.popleft()
        
        if vertex == goal:
            path = []
            while vertex is not None:
                path.append(vertex)
                vertex = parent[vertex]
            return path[::-1]

        for neighbor in graph.neighbors(vertex):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = vertex
                queue.append(neighbor)
                
    return None  # Если путь не найден


#Алгоритм Дейкстры
import heapq

def dijkstra(graph, start, goal):
    # Начальные расстояния до всех вершин — бесконечность
    distances = {vertex: float('infinity') for vertex in graph.nodes()}
    distances[start] = 0
    # Инициализация приоритетной очереди
    priority_queue = [(0, start)]
    parent = {start: None}

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        # Если нашли целевую вершину, возвращаем путь
        if current_vertex == goal:
            path = []
            while current_vertex is not None:
                path.append(current_vertex)
                current_vertex = parent[current_vertex]
            return path[::-1]

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, attributes in graph[current_vertex].items():
            weight = attributes.get('weight', 1)  # Если веса нет, считаем его равным 1
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                parent[neighbor] = current_vertex
                heapq.heappush(priority_queue, (distance, neighbor))

    return None  # Если путь не найден



#Алгоритм Беллмана-Форда
def bellman_ford(graph, start, goal):
    # Начальные расстояния до всех вершин — бесконечность
    distances = {vertex: float('infinity') for vertex in graph.nodes()}
    distances[start] = 0
    parent = {start: None}

    # Повторяем итерации |V|-1 раз, где V — количество вершин
    for _ in range(len(graph.nodes()) - 1):
        for vertex in graph.nodes():
            for neighbor, attributes in graph[vertex].items():
                weight = attributes.get('weight', 1)
                if distances[vertex] + weight < distances[neighbor]:
                    distances[neighbor] = distances[vertex] + weight
                    parent[neighbor] = vertex

    # Проверка на отрицательные циклы
    for vertex in graph.nodes():
        for neighbor, attributes in graph[vertex].items():
            weight = attributes.get('weight', 1)
            if distances[vertex] + weight < distances[neighbor]:
                raise ValueError("Граф содержит отрицательный цикл")

    # Восстановление пути, если найден
    path = []
    current_vertex = goal
    while current_vertex is not None:
        path.append(current_vertex)
        current_vertex = parent[current_vertex]
    return path[::-1] if path[0] == start else None


#Алгоритм A-star
import math

def heuristic(a, b):
    # Пример эвристики: Евклидово расстояние (подходит для графа с координатами)
    (x1, y1) = a
    (x2, y2) = b
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def a_star(graph, start, goal):
    open_set = [(0, start)]
    g_score = {vertex: float('infinity') for vertex in graph.nodes()}
    g_score[start] = 0
    f_score = {vertex: float('infinity') for vertex in graph.nodes()}
    f_score[start] = heuristic(start, goal)
    parent = {start: None}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]

        for neighbor, attributes in graph[current].items():
            weight = attributes.get('weight', 1)
            tentative_g_score = g_score[current] + weight

            if tentative_g_score < g_score[neighbor]:
                parent[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # Если путь не найден

#Тестирование
import networkx as nx

# Создаем взвешенный граф
G = nx.Graph()
G.add_edge('A', 'B', weight=1)
G.add_edge('A', 'C', weight=4)
G.add_edge('B', 'C', weight=2)
G.add_edge('B', 'D', weight=5)
G.add_edge('C', 'D', weight=1)

# Тестируем алгоритмы
start = 'A'
goal = 'D'

print("BFS:", bfs(G, start, goal))
print("Dijkstra:", dijkstra(G, start, goal))
print("Bellman-Ford:", bellman_ford(G, start, goal))
print("A*:", a_star(G, start, goal))





