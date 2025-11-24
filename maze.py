import pygame
import random
import time
from collections import deque
import heapq

# COLORES
TILE =40 
WHITE = (255,255,255)
BLACK = (0,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
YELLOW = (255,255,0)
SIZE = 16

pygame.init()
screen = pygame.display.set_mode((800, 800))
pygame.display.set_caption("Laberinto - BFS vs A* (Diferencias)")

def generate_maze(n, m):
    maze = [["0" if random.random() > 0.25 else "1" for _ in range(m)] for _ in range(n)]
    maze[0][0] = "S"
    maze[n-1][m-1] = "E"
    return maze

maze = generate_maze(SIZE, SIZE)

def draw_maze():
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            color = WHITE
            if cell == "1": color = BLACK
            elif cell == "S": color = GREEN
            elif cell == "E": color = BLUE
            pygame.draw.rect(screen, color, (j*TILE, i*TILE, TILE, TILE))

def find_start_end(maze):
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 'S':
                start = (i, j)
            elif maze[i][j] == 'E':
                end = (i, j)
    return start, end

def bfs(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    directions = [(0,1), (1,0), (0,-1), (-1,0)]
    queue = deque([(start, [start])])
    visited = set([start])
    nodes_explored = 0
    start_time = time.time()
    
    while queue:
        (current, path) = queue.popleft()
        nodes_explored += 1
        
        if current == end:
            return path, nodes_explored, time.time()-start_time, True
        
        for dx,dy in directions:
            nx, ny = current[0]+dx, current[1]+dy
            if (0<=nx<rows and 0<=ny<cols and maze[nx][ny]!='1' and (nx,ny) not in visited):
                visited.add((nx,ny))
                queue.append(((nx,ny), path+[(nx,ny)]))
    
    return None, nodes_explored, time.time()-start_time, False

def a_star(maze, start, end):
    rows, cols = len(maze), len(maze[0])
    directions = [(0,1), (1,0), (0,-1), (-1,0)]
    
    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    nodes_explored = 0
    start_time = time.time()
    
    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_explored += 1
        
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, nodes_explored, time.time()-start_time, True
        
        for dx,dy in directions:
            nx, ny = current[0]+dx, current[1]+dy
            neighbor = (nx, ny)
            
            if 0<=nx<rows and 0<=ny<cols and maze[nx][ny]!='1':
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None, nodes_explored, time.time()-start_time, False

# Variables para control
path_bfs = []
path_astar = []

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_r:
                maze = generate_maze(SIZE, SIZE)
                path_bfs = []
                path_astar = []
            if e.key == pygame.K_SPACE:
                start, end = find_start_end(maze)
                path_bfs, nodes_bfs, time_bfs, success_bfs = bfs(maze, start, end)
                path_astar, nodes_astar, time_astar, success_astar = a_star(maze, start, end)
                
                print("=== RESULTADOS ===")
                print(f"BFS: Tiempo={time_bfs:.4f}s, Nodos={nodes_bfs}, Longitud={len(path_bfs) if success_bfs else 'No solution'}")
                print(f"A*:  Tiempo={time_astar:.4f}s, Nodos={nodes_astar}, Longitud={len(path_astar) if success_astar else 'No solution'}")

    # DIBUJAR TODO EN ORDEN CORRECTO
    screen.fill(WHITE)  # Limpiar pantalla primero
    draw_maze()  # Dibujar laberinto base
    
    # SOLO dibujar caminos si existen
    if path_bfs and path_astar:
        set_bfs = set(path_bfs)
        set_astar = set(path_astar)
        
        # SOLO BFS (AZUL)
        for (i,j) in set_bfs - set_astar:
            if maze[i][j] not in ['S','E']:
                pygame.draw.rect(screen, BLUE, (j*TILE, i*TILE, TILE, TILE))
        
        # SOLO A* (AMARILLO)
        for (i,j) in set_astar - set_bfs:
            if maze[i][j] not in ['S','E']:
                pygame.draw.rect(screen, YELLOW, (j*TILE, i*TILE, TILE, TILE))
        
        # AMBOS (VERDE)
        for (i,j) in set_bfs & set_astar:
            if maze[i][j] not in ['S','E']:
                pygame.draw.rect(screen, GREEN, (j*TILE, i*TILE, TILE, TILE))
    
    pygame.display.flip()

pygame.quit()