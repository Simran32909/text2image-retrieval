from queue import PriorityQueue, Queue

graph = {
    'Arad': [('Zerind', 75), ('Timisoara', 118), ('Sibiu', 140)],
    'Bucharest': [('Fagaras', 211), ('Pitesti', 101), ('Urziceni', 85), ('Giurgiu', 90)],
    'Craiova': [('Drobeta', 120), ('Pitesti', 138), ('Rimnicu', 146)],
    'Drobeta': [('Mehadia', 75), ('Craiova', 120)],
    'Eforie': [('Hirsova', 86)],
    'Fagaras': [('Sibiu', 99), ('Bucharest', 211)],
    'Giurgiu': [('Bucharest', 90)],
    'Hirsova': [('Urziceni', 98), ('Eforie', 86)],
    'Iasi': [('Neamt', 87), ('Vaslui', 92)],
    'Lugoj': [('Timisoara', 111), ('Mehadia', 70)],
    'Mehadia': [('Drobeta', 75), ('Lugoj', 70)],
    'Neamt': [('Iasi', 87)],
    'Oradea': [('Zerind', 71), ('Sibiu', 151)],
    'Pitesti': [('Rimnicu', 97), ('Craiova', 138), ('Bucharest', 101)],
    'Rimnicu': [('Sibiu', 80), ('Pitesti', 97), ('Craiova', 146)],
    'Sibiu': [('Arad', 140), ('Oradea', 151), ('Fagaras', 99), ('Rimnicu', 80)],
    'Timisoara': [('Arad', 118), ('Lugoj', 111)],
    'Urziceni': [('Bucharest', 85), ('Hirsova', 98), ('Vaslui', 142)],
    'Vaslui': [('Iasi', 92), ('Urziceni', 142)],
    'Zerind': [('Oradea', 71), ('Arad', 75)]
}


def bfs(start, goal):
    queue = Queue()
    queue.put((start, [start]))
    while not queue.empty():
        node, path = queue.get()
        if node == goal:
            return path
        for neighbor, _ in graph[node]:
            if neighbor not in path:
                queue.put((neighbor, path + [neighbor]))


def dfs(start, goal, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []
    visited.add(start)
    path.append(start)
    if start == goal:
        return path
    for neighbor, _ in graph[start]:
        if neighbor not in visited:
            result = dfs(neighbor, goal, visited, path)
            if result:
                return result
    path.pop()
    return None


def ucs(start, goal):
    pq = PriorityQueue()
    pq.put((0, start, [start]))
    visited = set()
    while not pq.empty():
        cost, node, path = pq.get()
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            return path
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                pq.put((cost + weight, neighbor, path + [neighbor]))


def depth_limited_search(start, goal, limit):
    def dls(node, path, depth):
        if depth > limit:
            return None
        if node == goal:
            return path
        for neighbor, _ in graph[node]:
            if neighbor not in path:
                result = dls(neighbor, path + [neighbor], depth + 1)
                if result:
                    return result
        return None

    return dls(start, [start], 0)


def iterative_deepening_search(start, goal, max_depth):
    for limit in range(max_depth + 1):
        result = depth_limited_search(start, goal, limit)
        if result:
            return result
    return None


# Test the algorithms
start, goal = 'Arad', 'Bucharest'
print("BFS:", bfs(start, goal))
print("DFS:", dfs(start, goal))
print("UCS:", ucs(start, goal))
print("DLS (limit=3):", depth_limited_search(start, goal, 3))
print("IDS (max_depth=10):", iterative_deepening_search(start,goal,10))