import numpy as np
import cv2
import sys
import os

# --- Configuração ---
sys.path.append(os.path.abspath("Python"))
try:
    from gapy.ga import gago
    from gapy.utilities import bits2bytes
except ImportError:
    print("ERRO: Biblioteca 'gapy' não encontrada.")
    sys.exit(1)

# --- 1. Carregamento ---
IMAGE_FILE = "ImagemDemo.png"
if not os.path.exists(IMAGE_FILE):
    from gerador_imagem import create_demo_image
    create_demo_image(IMAGE_FILE)

TARGET_IMG = cv2.imread(IMAGE_FILE, cv2.IMREAD_GRAYSCALE)
H, W = TARGET_IMG.shape

# --- 2. CÉREBRO COMPLEXO (Navegação + Reconhecimento) ---
class SmartCreatureRNA:
    def __init__(self):
        # Entradas: 2 de Navegação (Gradiente X, Y) + 8 de Visão (Bigodes)
        self.input_size = 10 
        self.hidden_size = 12 # Cérebro maior para processar geometria
        # Saídas: Move X, Move Y, MORDER (Gatilho)
        self.output_size = 3 
        
        self.w1_shape = (self.input_size + 1, self.hidden_size)
        self.w2_shape = (self.hidden_size + 1, self.output_size)
        self.total_weights = (self.w1_shape[0] * self.w1_shape[1]) + \
                             (self.w2_shape[0] * self.w2_shape[1])

    def get_chromosome_size(self):
        return self.total_weights * 8

    def decode_weights(self, bits):
        bytes_data = bits2bytes(bits, dtype='int8')
        weights_normalized = bytes_data.astype(np.float32) / 10.0
        split_idx = self.w1_shape[0] * self.w1_shape[1]
        W1 = weights_normalized[:split_idx].reshape(self.w1_shape)
        W2 = weights_normalized[split_idx:].reshape(self.w2_shape)
        return W1, W2

    def forward(self, sensors, W1, W2):
        inputs = np.array(sensors)
        x_in = np.append(inputs, 1.0)
        h = np.tanh(np.dot(x_in, W1))
        h_in = np.append(h, 1.0)
        output = np.tanh(np.dot(h_in, W2))
        return output

rna_struct = SmartCreatureRNA()

# --- 3. SISTEMA SENSORIAL AVANÇADO ---
def get_advanced_sensors(px, py, dist_map, img):
    h, w = dist_map.shape
    ix, iy = int(px), int(py)
    
    # 1. Sensores de Navegação (Cheiro)
    # Garante que não sai da matriz
    ix = np.clip(ix, 2, w-3)
    iy = np.clip(iy, 2, h-3)
    
    gx = (dist_map[iy, ix+2] - dist_map[iy, ix-2]) / 2.0
    gy = (dist_map[iy+2, ix] - dist_map[iy-2, ix]) / 2.0
    
    # 2. Sensores de Visão/Tato (Bigodes)
    # Mede a distância até a parede branca em 8 direções
    whiskers = []
    directions = [
        (0,-1), (1,-1), (1,0), (1,1),  # N, NE, L, SE
        (0,1), (-1,1), (-1,0), (-1,-1) # S, SO, O, NO
    ]
    
    max_reach = 30.0 # Visão máxima
    
    for dx, dy in directions:
        dist = max_reach
        for r in range(1, int(max_reach)):
            nx, ny = int(px + dx*r), int(py + dy*r)
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                dist = r; break
            if img[ny, nx] > 100: # Bateu no branco
                dist = r; break
        
        # Normaliza entre -1 e 1 para a RNA
        whiskers.append((dist / max_reach) * 2 - 1)
        
    # Junta tudo: [GradX, GradY, W1, W2, ..., W8]
    return [gx, gy] + whiskers

# --- 4. AVALIADOR DE "MORDIDA" (O Juiz da Realidade) ---
def evaluate_bite(cx, cy, img):
    """
    Esta função representa a 'Física do Mundo'.
    Se a criatura morder, esta função diz se ela pegou algo ou não.
    """
    h, w = img.shape
    if cx < 0 or cx >= w or cy < 0 or cy >= h: return 0
    
    # Se morder no branco -> Punição
    if img[cy, cx] > 100: return -1 # Erro (Morde Vazio)
    
    # Se morder no preto -> Verifica se é uma bola válida
    # Expande para ver o tamanho real
    radius = 0
    for r in range(2, 50):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, 1)
        mean_val = cv2.mean(img, mask=mask)[0]
        if mean_val > 50: break
        radius = r
    
    # Se for uma bola válida (raio > 5) -> Sucesso
    if radius >= 5: return radius
    
    return -0.5 # Punição pequena (Morde sujeira)

# --- 5. TREINAMENTO (APRENDIZADO REFORÇADO) ---
_, binary_static = cv2.threshold(TARGET_IMG, 127, 255, cv2.THRESH_BINARY)
DIST_MAP_STATIC = cv2.distanceTransform(binary_static, cv2.DIST_L2, 5)

def fitness_function(individual_bits):
    try: W1, W2 = rna_struct.decode_weights(individual_bits)
    except: return 1e6
    
    num_agents = 8
    steps = 40
    score = 0
    
    # Cenários de teste aleatórios
    np.random.seed(42)
    start_states = np.random.rand(num_agents, 2)
    
    for state in start_states:
        curr = state.copy()
        for _ in range(steps):
            px, py = int(curr[0]*(W-1)), int(curr[1]*(H-1))
            
            # Sensores
            sensors = get_advanced_sensors(px, py, DIST_MAP_STATIC, TARGET_IMG)
            
            # Cérebro Decide
            output = rna_struct.forward(sensors, W1, W2)
            dx, dy = output[0], output[1]
            bite_trigger = output[2] # > 0 significa MORDER
            
            # Ação de Movimento
            curr += [dx*0.05, dy*0.05]
            curr = np.clip(curr, 0, 1)
            
            # Custo de Energia por movimento (incentiva rapidez)
            score -= 1
            
            # Ação de Mordida (Decisão da IA)
            if bite_trigger > 0.5:
                result = evaluate_bite(px, py, TARGET_IMG)
                
                if result > 0: # Acertou bola!
                    score += 500 # Recompensa Enorme
                    # Para este agente, missão cumprida
                    break 
                elif result == -1: # Mordeu vento
                    score -= 50 # Punição Severa
                else:
                    score -= 10 # Mordeu sujeira
                    
    # Queremos MAXIMIZAR score, mas o GA minimiza, então retornamos negativo
    return -score / num_agents

# --- 6. EXECUÇÃO ---
def run_learning_creature(best_bits):
    print("\n--- CRIATURA QUE APRENDEU A MORDER ---")
    W1, W2 = rna_struct.decode_weights(best_bits)
    
    final_img = cv2.cvtColor(TARGET_IMG, cv2.COLOR_GRAY2BGR)
    mental_map = TARGET_IMG.copy()
    
    # Função auxiliar para achar comida (apenas para teleporte inicial)
    def find_food(mmap):
        k = np.ones((3,3), np.uint8)
        cln = cv2.erode(mmap, k, iterations=1)
        pts = cv2.findNonZero(cv2.threshold(cln, 100, 255, cv2.THRESH_BINARY_INV)[1])
        return pts
    
    food = find_food(mental_map)
    if food is None: return
    curr = np.array([food[0][0][0]/W, food[0][0][1]/H])
    
    _, bw = cv2.threshold(mental_map, 127, 255, cv2.THRESH_BINARY)
    curr_dist_map = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    
    video_file = 'aprendizado_bola.avi'
    out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (W, H))
    
    found_count = 0
    patience = 0
    
    loop = 0
    while True:
        loop += 1
        if loop > 20000: break
        
        px, py = int(curr[0]*(W-1)), int(curr[1]*(H-1))
        
        # 1. Visualização
        frame = final_img.copy()
        # Desenha os bigodes (visualizar o que ela vê)
        sensors = get_advanced_sensors(px, py, curr_dist_map, TARGET_IMG)
        whiskers = sensors[2:] # 8 ultimos são bigodes
        dirs = [(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)]
        
        for i, w_val in enumerate(whiskers):
            # Desnormaliza (-1..1 -> 0..30)
            dist_vis = (w_val + 1) / 2 * 30
            dx, dy = dirs[i]
            ex, ey = int(px + dx*dist_vis), int(py + dy*dist_vis)
            cv2.line(frame, (px, py), (ex, ey), (0, 255, 255), 1)
            
        cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)
        out.write(frame)
        
        # 2. Cérebro
        output = rna_struct.forward(sensors, W1, W2)
        dx, dy, bite = output
        
        # 3. Ação
        curr += [dx*0.04, dy*0.04]
        curr = np.clip(curr, 0, 1)
        
        # 4. Decisão de Morder
        if bite > 0.5:
            # Verifica realidade
            radius = evaluate_bite(px, py, TARGET_IMG)
            
            if radius > 0: # ACERTOU!
                # Confere se é nova
                is_new = True
                # (Lógica simples de duplicata visual para o vídeo)
                if mental_map[py, px] > 100: is_new = False

                if is_new:
                    found_count += 1
                    print(f"MORDIDA CERTA! Bola #{found_count} R={radius}")
                    
                    cv2.circle(final_img, (px, py), radius, (0, 255, 0), 2)
                    cv2.circle(mental_map, (px, py), int(radius*1.4), 255, -1)
                    
                    # Atualiza mundo
                    _, bw = cv2.threshold(mental_map, 127, 255, cv2.THRESH_BINARY)
                    curr_dist_map = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
                    
                    # Flash Visual
                    for _ in range(5): 
                        cv2.circle(frame, (px, py), radius, (0, 0, 255), 3)
                        out.write(frame)
                    
                    patience = 1000 # Força teleporte
            else:
                # Errou a mordida (mordeu o chão ou borda)
                # Visualmente mostramos um X vermelho pequeno
                cv2.line(frame, (px-2, py-2), (px+2, py+2), (0,0,255), 1)
                cv2.line(frame, (px-2, py+2), (px+2, py-2), (0,0,255), 1)
                out.write(frame)

        # Gestão de Fome/Paciência
        patience += 1
        if patience > 150:
            food = find_food(mental_map)
            if food is None or len(food) == 0:
                print("--- CAÇADA COMPLETA ---")
                for _ in range(30): out.write(final_img)
                break
            
            idx = np.random.randint(len(food))
            tx, ty = food[idx][0]
            # Offset para não cair no centro (obrigando ela a se ajustar)
            off = np.random.uniform(-0.02, 0.02, 2)
            curr = np.array([tx/W, ty/H]) + off
            curr = np.clip(curr, 0, 1)
            patience = 0

    out.release()
    cv2.imwrite("resultado_aprendizado.png", final_img)
    print("Salvo: resultado_aprendizado.png e aprendizado_bola.avi")

def run():
    print("1. Evoluindo Cérebro (Isso pode demorar um pouco)...")
    # Aumentamos a população pois o problema é mais difícil
    nbits = rna_struct.get_chromosome_size()
    options = {"PopulationSize": 100, "Generations": 80, "MutationFcn": 0.15, "EliteCount": 4}
    
    best_bits, _, fit = gago(fitness_function, nbits, options)
    print(f"   Fitness Final: {-fit[0]:.2f} (Quanto maior, melhor)")
    
    np.save("melhor_rna_inteligente.npy", best_bits)
    run_learning_creature(best_bits)

if __name__ == "__main__":
    run()