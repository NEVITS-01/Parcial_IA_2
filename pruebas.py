import random
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import heapq
from datetime import datetime

# CONFIGURACIÓN IDÉNTICA a maze.py
SIZE = 16  # MISMO tamaño
NUM_EXPERIMENTOS = 20

# MÉTODOS EXACTAMENTE IGUALES a maze.py
def generate_maze(n, m):
    # MISMA probabilidad de paredes (25%)
    maze = [["0" if random.random() > 0.25 else "1" for _ in range(m)] for _ in range(n)]
    maze[0][0] = "S"
    maze[n-1][m-1] = "E"
    return maze

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

def ejecutar_experimentos():
    """Ejecuta experimentos"""
    resultados = []
    
    print("🔬 Ejecutando 20 experimentos ")
    print(f"📏 Configuración: Laberintos {SIZE}x{SIZE}, 25% de paredes")
    
    for i in range(NUM_EXPERIMENTOS):
        # MISMO laberinto que generaría maze.py
        laberinto = generate_maze(SIZE, SIZE)
        inicio, fin = find_start_end(laberinto)
        
        # MISMO BFS que usaría maze.py
        camino_bfs, nodos_bfs, tiempo_bfs, exito_bfs = bfs(laberinto, inicio, fin)
        
        # MISMO A* que usaría maze.py
        camino_astar, nodos_astar, tiempo_astar, exito_astar = a_star(laberinto, inicio, fin)
        
        resultados.append({
            'experimento': i + 1,
            'tamaño': f"{SIZE}x{SIZE}",
            'bfs_tiempo': tiempo_bfs,
            'bfs_nodos': nodos_bfs,
            'bfs_longitud': len(camino_bfs) if exito_bfs else 0,
            'bfs_exito': exito_bfs,
            'astar_tiempo': tiempo_astar,
            'astar_nodos': nodos_astar,
            'astar_longitud': len(camino_astar) if exito_astar else 0,
            'astar_exito': exito_astar
        })
        
        print(f"✅ Experimento {i+1:2d}: BFS={tiempo_bfs:.4f}s, A*={tiempo_astar:.4f}s")
    
    return resultados

# El resto del código (guardar CSV, gráficas, estadísticas)

def guardar_resultados_csv(resultados):
    nombre_archivo = f"resultados_laberintos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(nombre_archivo, 'w', newline='') as csvfile:
        campo_nombres = ['experimento', 'tamaño', 'bfs_tiempo', 'bfs_nodos', 'bfs_longitud', 'bfs_exito',
                        'astar_tiempo', 'astar_nodos', 'astar_longitud', 'astar_exito']
        writer = csv.DictWriter(csvfile, fieldnames=campo_nombres)
        
        writer.writeheader()
        for resultado in resultados:
            writer.writerow(resultado)
    
    print(f"💾 Resultados guardados en: {nombre_archivo}")
    return nombre_archivo

def calcular_estadisticas(resultados):
    exitos_bfs = [r for r in resultados if r['bfs_exito']]
    exitos_astar = [r for r in resultados if r['astar_exito']]
    
    stats = {
        'tasa_exito_bfs': len(exitos_bfs) / NUM_EXPERIMENTOS * 100,
        'tasa_exito_astar': len(exitos_astar) / NUM_EXPERIMENTOS * 100,
        
        'tiempo_promedio_bfs': np.mean([r['bfs_tiempo'] for r in exitos_bfs]) if exitos_bfs else 0,
        'tiempo_promedio_astar': np.mean([r['astar_tiempo'] for r in exitos_astar]) if exitos_astar else 0,
        
        'nodos_promedio_bfs': np.mean([r['bfs_nodos'] for r in exitos_bfs]) if exitos_bfs else 0,
        'nodos_promedio_astar': np.mean([r['astar_nodos'] for r in exitos_astar]) if exitos_astar else 0,
        
        'longitud_promedio_bfs': np.mean([r['bfs_longitud'] for r in exitos_bfs]) if exitos_bfs else 0,
        'longitud_promedio_astar': np.mean([r['astar_longitud'] for r in exitos_astar]) if exitos_astar else 0,
    }
    
    return stats

def generar_graficas(resultados, stats):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('COMPARATIVA BFS vs A* - 20 EXPERIMENTOS', fontsize=14, fontweight='bold')
    
    experimentos = [r['experimento'] for r in resultados]
    tiempos_bfs = [r['bfs_tiempo'] for r in resultados]
    tiempos_astar = [r['astar_tiempo'] for r in resultados]
    nodos_bfs = [r['bfs_nodos'] for r in resultados]
    nodos_astar = [r['astar_nodos'] for r in resultados]
    
    # 1. Gráfica de tiempos
    ax1.plot(experimentos, tiempos_bfs, 'b-', marker='o', linewidth=2, markersize=4, label='BFS')
    ax1.plot(experimentos, tiempos_astar, 'r-', marker='s', linewidth=2, markersize=4, label='A*')

    ax1.set_ylabel('Tiempo (segundos)')
    ax1.set_title('Tiempo de Ejecución')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Gráfica de nodos explorados
    ax2.plot(experimentos, nodos_bfs, 'b-', marker='o', linewidth=2, markersize=4, label='BFS')
    ax2.plot(experimentos, nodos_astar, 'r-', marker='s', linewidth=2, markersize=4, label='A*')
    ax2.set_ylabel('Nodos Explorados')
    ax2.set_title('Eficiencia en Exploración')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Gráfica de barras comparativas
    categorias = ['Tiempo (s)', 'Nodos', 'Longitud']
    valores_bfs = [stats['tiempo_promedio_bfs'], stats['nodos_promedio_bfs'], stats['longitud_promedio_bfs']]
    valores_astar = [stats['tiempo_promedio_astar'], stats['nodos_promedio_astar'], stats['longitud_promedio_astar']]
    
    x = np.arange(len(categorias))
    width = 0.35
    
    ax3.bar(x - width/2, valores_bfs, width, label='BFS', color='blue', alpha=0.7)
    ax3.bar(x + width/2, valores_astar, width, label='A*', color='red', alpha=0.7)
    ax3.set_xlabel('Métricas')
    ax3.set_ylabel('Valor Promedio')
    ax3.set_title('Comparación de Métricas Promedio')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categorias)
    ax3.legend()
    
    # 4. Gráfica de tasas de éxito
    tasas = [stats['tasa_exito_bfs'], stats['tasa_exito_astar']]
    algoritmos = ['BFS', 'A*']
    
    barras = ax4.bar(algoritmos, tasas, color=['blue', 'red'], alpha=0.7)
    ax4.set_ylabel('Tasa de Éxito (%)')
    ax4.set_title('Tasa de Éxito')
    ax4.set_ylim(0, 100)
    
    for barra, tasa in zip(barras, tasas):
        height = barra.get_height()
        ax4.text(barra.get_x() + barra.get_width()/2., height + 1,
                f'{tasa:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('comparacion_bfs_vs_astar.png', dpi=300, bbox_inches='tight')
    plt.show()

def mostrar_estadisticas(resultados, stats):
    print("\n" + "="*70)
    print("📊 RESULTADOS ESTADÍSTICOS - 20 EXPERIMENTOS")
    print("="*70)
    
    print(f"\n📈 MÉTRICAS PRINCIPALES:")
    print(f"   BFS  - Tasa de éxito: {stats['tasa_exito_bfs']:6.1f}%")
    print(f"   A*   - Tasa de éxito: {stats['tasa_exito_astar']:6.1f}%")
    print(f"   BFS  - Tiempo promedio: {stats['tiempo_promedio_bfs']:8.4f} segundos")
    print(f"   A*   - Tiempo promedio: {stats['tiempo_promedio_astar']:8.4f} segundos")
    print(f"   BFS  - Nodos promedio: {stats['nodos_promedio_bfs']:8.1f} nodos")
    print(f"   A*   - Nodos promedio: {stats['nodos_promedio_astar']:8.1f} nodos")
    print(f"   BFS  - Longitud promedio: {stats['longitud_promedio_bfs']:6.1f} pasos")
    print(f"   A*   - Longitud promedio: {stats['longitud_promedio_astar']:6.1f} pasos")
    
    # Análisis de por qué A* puede ser más lento en problemas pequeños
    print(f"\n🔍 ANÁLISIS:")
    print(f"   • A* explora {((stats['nodos_promedio_bfs'] - stats['nodos_promedio_astar'])/stats['nodos_promedio_bfs'])*100:+.1f}% menos nodos que BFS")
    print(f"   • En problemas pequeños (16x16), el overhead de A* (heap + heurística)")
    print(f"     puede hacerlo más lento a pesar de explorar menos nodos")
    print(f"   • En problemas más grandes, A* generalmente supera a BFS en velocidad")

if __name__ == "__main__":
    print("🧪 INICIANDO EXPERIMENTOS: BFS vs A*")
    print("="*60)
    print("✅ Laberintos: 16x16 con 25% de paredes")
    print("✅ Algoritmos: BFS y A* idénticos a la versión visual")
    
    resultados = ejecutar_experimentos()
    archivo_csv = guardar_resultados_csv(resultados)
    stats = calcular_estadisticas(resultados)
    mostrar_estadisticas(resultados, stats)
    
    print("\n📊 Generando gráficas comparativas...")
    generar_graficas(resultados, stats)
    
    print(f"\n✅ ANÁLISIS COMPLETADO!")
    print(f"   • Archivo CSV: {archivo_csv}")
    print(f"   • Gráficas: comparacion_bfs_vs_astar.png")