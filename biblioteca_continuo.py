import numpy as np
import matplotlib.pyplot as plt

# Necessário para plotagem 3D
from mpl_toolkits.mplot3d import Axes3D 

# =========================================================
# FUNÇÕES OBJETIVO (f1 a f6)
# =========================================================

def f1(x): 
    return x[0]**2 + x[1]**2

def f2(x):
    term1 = np.exp(-(x[0]**2 + x[1]**2))
    term2 = 2 * np.exp(-((x[0]-1.7)**2 + (x[1]-1.7)**2))
    return term1 + term2

def f3(x):
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2)))
    term2 = -np.exp(0.5 * (np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1])))
    return term1 + term2 + 20 + np.e

def f4(x):
    term1 = x[0]**2 - 10 * np.cos(2*np.pi*x[0]) + 10
    term2 = x[1]**2 - 10 * np.cos(2*np.pi*x[1]) + 10
    return term1 + term2

def f5(x):
    term1 = (x[0] * np.cos(x[0])) / 20.0
    term2 = 2 * np.exp(-(x[0]**2 + (x[1]-1)**2))
    term3 = 0.01 * x[0] * x[1]
    return term1 + term2 + term3

def f6(x):
    return x[0] * np.sin(4*np.pi*x[0]) - x[1] * np.sin(4*np.pi*x[1] + np.pi) + 1

PROBLEMAS = {
    "f1": {"func": f1, "bounds": [(-100, 100), (-100, 100)], "tipo": "min"},
    "f2": {"func": f2, "bounds": [(-2, 4), (-2, 5)], "tipo": "max"},
    "f3": {"func": f3, "bounds": [(-8, 8), (-8, 8)], "tipo": "min"},
    "f4": {"func": f4, "bounds": [(-5.12, 5.12), (-5.12, 5.12)], "tipo": "min"},
    "f5": {"func": f5, "bounds": [(-10, 10), (-10, 10)], "tipo": "max"},
    "f6": {"func": f6, "bounds": [(-1, 3), (-1, 3)], "tipo": "max"},
}

# =========================================================
# ALGORITMOS DE BUSCA
# =========================================================

def verifica_limites(x, bounds):
    x_clippado = []
    for i in range(len(x)):
        x_clippado.append(np.clip(x[i], bounds[i][0], bounds[i][1]))
    return np.array(x_clippado)

def melhor_que(cand, atual, tipo):
    return cand < atual if tipo == "min" else cand > atual

# Hill Climbing
def hill_climbing(func, bounds, tipo, max_iter=1000, eps=0.1):
    # Início no limite inferior
    x_atual = np.array([bounds[0][0], bounds[1][0]]) 
    val_atual = func(x_atual)
    sem_melhora = 0
    
    for _ in range(max_iter):
        vizinho = x_atual + np.random.uniform(-eps, eps, size=2)
        vizinho = verifica_limites(vizinho, bounds)
        val_vizinho = func(vizinho)

        if melhor_que(val_vizinho, val_atual, tipo):
            x_atual = vizinho
            val_atual = val_vizinho
            sem_melhora = 0
        else:
            sem_melhora += 1
        if sem_melhora >= 50: break 
    return x_atual, val_atual

# Local Random Search (LRS)
def local_random_search(func, bounds, tipo, max_iter=1000, sigma=0.5):
    # Início aleatório
    x_atual = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
    val_atual = func(x_atual)
    sem_melhora = 0

    for _ in range(max_iter):
        # Perturbação gaussiana
        vizinho = x_atual + np.random.normal(0, sigma, size=2)
        vizinho = verifica_limites(vizinho, bounds)
        val_vizinho = func(vizinho)

        if melhor_que(val_vizinho, val_atual, tipo):
            x_atual = vizinho
            val_atual = val_vizinho
            sem_melhora = 0
        else:
            sem_melhora += 1
        if sem_melhora >= 50: break
    return x_atual, val_atual

# Global Random Search (GRS)
def global_random_search(func, bounds, tipo, max_iter=1000):
    x_atual = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
    val_atual = func(x_atual)
    sem_melhora = 0

    for _ in range(max_iter):
        # Candidato totalmente aleatório
        cand = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        val_cand = func(cand)

        if melhor_que(val_cand, val_atual, tipo):
            x_atual = cand
            val_atual = val_cand
            sem_melhora = 0
        else:
            sem_melhora += 1
        if sem_melhora >= 50: break
    return x_atual, val_atual

# =========================================================
# FUNÇÃO DE PLOTAGEM (2D e 3D)
# =========================================================

def plotar_distribuicao_algoritmo(func, bounds, solucoes, melhor_global, titulo):
    """
    Gera dois gráficos lado a lado: 2D (Contorno) e 3D (Superfície).
    """
    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]
    
    # Cria malha para o terreno (resolução 50x50 para ser rápido)
    x = np.linspace(xmin, xmax, 50)
    y = np.linspace(ymin, ymax, 50)
    X, Y = np.meshgrid(x, y)
    
    # Calcula Z para a superfície
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])
    
    # Prepara dados dos pontos (soluções)
    xs_red = [s[0] for s in solucoes]
    ys_red = [s[1] for s in solucoes]
    # Calcula a altura Z de cada solução encontrada para plotar no 3D
    zs_red = [func(s) for s in solucoes]
    
    # Prepara dados do melhor global (ponto verde)
    gx, gy = melhor_global
    gz = func(melhor_global)

    # Configura a figura com 2 subplots
    fig = plt.figure(figsize=(14, 6))
    
    # --- PLOT 1: 2D Contour ---
    ax1 = fig.add_subplot(1, 2, 1)
    contour = ax1.contourf(X, Y, Z, levels=25, cmap='viridis')
    plt.colorbar(contour, ax=ax1, label='Valor f(x)')
    
    # Pontos vermelhos (todas as rodadas)
    ax1.scatter(xs_red, ys_red, c='red', marker='x', s=40, label='Rodadas', alpha=0.7)
    # Ponto verde (melhor global)
    ax1.scatter(gx, gy, c='#00FF00', marker='X', s=150, linewidths=2, label='Melhor Global', edgecolors='black', zorder=10)
    
    ax1.set_title(f"{titulo} (Vista 2D)")
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.legend()
    
    # --- PLOT 2: 3D Surface ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # Superfície transparente
    ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5, edgecolor='none')
    
    # Pontos vermelhos flutuando no 3D
    # Adicionamos um pequeno valor ao Z para garantir que o ponto não fica escondido "dentro" do chão
    offset = (np.max(Z) - np.min(Z)) * 0.01 
    ax2.scatter(xs_red, ys_red, np.array(zs_red) + offset, c='red', marker='x', s=40, label='Rodadas')
    
    # Ponto verde flutuando no 3D
    ax2.scatter(gx, gy, gz + offset, c='#00FF00', marker='X', s=200, label='Melhor Global', edgecolors='black')
    
    ax2.set_title(f"{titulo} (Vista 3D)")
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('f(x)')
    
    plt.tight_layout()
    plt.show()

# =========================================================
# FUNÇÃO PRINCIPAL DA PARTE 1
# =========================================================

def resolver_parte_1(num_rodadas=100):
    print(f"\n=== PARTE 1: FUNÇÕES CONTÍNUAS ({num_rodadas} RODADAS) ===")
    
    print(f"{'Função':<6} | {'Algoritmo':<5} | {'Melhor Global':<15} | {'Média Final':<15}")
    print("-" * 55)

    for nome, dados in PROBLEMAS.items():
        func = dados["func"]
        bounds = dados["bounds"]
        tipo = dados["tipo"]
        
        # Ajuste dinâmico de parâmetros
        tam_dominio = bounds[0][1] - bounds[0][0]
        eps = tam_dominio * 0.05
        sigma = tam_dominio * 0.1

        dados_execucao = {
            "HC": {"vals": [], "coords": []},
            "LRS": {"vals": [], "coords": []},
            "GRS": {"vals": [], "coords": []}
        }
        
        # --- EXECUÇÃO DAS RODADAS ---
        for _ in range(num_rodadas):
            coord_hc, val_hc = hill_climbing(func, bounds, tipo, eps=eps)
            dados_execucao["HC"]["vals"].append(val_hc)
            dados_execucao["HC"]["coords"].append(coord_hc)
            
            coord_lrs, val_lrs = local_random_search(func, bounds, tipo, sigma=sigma)
            dados_execucao["LRS"]["vals"].append(val_lrs)
            dados_execucao["LRS"]["coords"].append(coord_lrs)
            
            coord_grs, val_grs = global_random_search(func, bounds, tipo)
            dados_execucao["GRS"]["vals"].append(val_grs)
            dados_execucao["GRS"]["coords"].append(coord_grs)

        # --- PROCESSAMENTO E PLOTAGEM ---
        for alg in ["HC", "LRS", "GRS"]:
            vals = dados_execucao[alg]["vals"]
            coords = dados_execucao[alg]["coords"]
            
            if tipo == "min":
                idx_best = np.argmin(vals)
                melhor_val = vals[idx_best]
            else:
                idx_best = np.argmax(vals)
                melhor_val = vals[idx_best]
            
            melhor_coord = coords[idx_best]
            media_val = np.mean(vals)
            
            print(f"{nome:<6} | {alg:<5} | {melhor_val:.4e}     | {media_val:.4e}")
            
            plotar_distribuicao_algoritmo(
                func, bounds, 
                solucoes=coords, 
                melhor_global=melhor_coord, 
                titulo=f"{nome} - {alg} ({num_rodadas} rodadas)"
            )