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

# --- PRÉ-PROCESSAMENTO ---
print("Calculando mapas...")
_, binary = cv2.threshold(TARGET_IMG, 127, 255, cv2.THRESH_BINARY)
# Mapa estático para treino
DIST_MAP_STATIC = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

# --- 2. RNA ---
class MaskRNA:
    def __init__(self):
        self.input_size = 4 
        self.hidden_size = 6
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

    def forward(self, state, sensors, W1, W2):
        r = state[2]
        gx, gy, dist = sensors
        inputs = np.array([gx, gy, np.tanh(dist / 40.0), r])
        x_in = np.append(inputs, 1.0)
        h = np.tanh(np.dot(x_in, W1))
        h_in = np.append(h, 1.0)
        output = np.tanh(np.dot(h_in, W2))
        return output

rna_struct = MaskRNA()

# --- 3. SENSORES ---
def get_sensors(px, py, dist_map):
    h, w = dist_map.shape
    px = int(np.clip(px, 2, w-3))
    py = int(np.clip(py, 2, h-3))
    gx = (dist_map[py, px+2] - dist_map[py, px-2]) / 2.0
    gy = (dist_map[py+2, px] - dist_map[py-2, px]) / 2.0
    dist = dist_map[py, px]
    return gx, gy, dist

# --- 4. AVALIAÇÃO (Contraste Puro) ---
def calculate_pure_contrast(cx, cy, r, img):
    h, w = img.shape
    if cx < 0 or cx >= w or cy < 0 or cy >= h: return -200.0
    if r < 4: return -200.0

    try:
        mask_in = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_in, (cx, cy), int(r * 0.8), 255, -1)
        mean_in = cv2.mean(img, mask=mask_in)[0]

        mask_out = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_out, (cx, cy), int(r + 3), 255, -1)
        cv2.circle(mask_out, (cx, cy), int(r), 0, -1)
        mean_out = cv2.mean(img, mask=mask_out)[0]

        contrast = mean_out - mean_in
        penalty = r * 0.05 
        return contrast - penalty
    except: return -200.0

def fitness_function(individual_bits):
    try:
        W1, W2 = rna_struct.decode_weights(individual_bits)
    except: return 1e6

    num_agents = 15
    steps = 40
    total_score = 0
    np.random.seed(42)
    start_states = np.random.rand(num_agents, 3)

    for state in start_states:
        curr = state.copy()
        for _ in range(steps):
            px = int(np.clip(curr[0]*(W-1), 0, W-1))
            py = int(np.clip(curr[1]*(H-1), 0, H-1))
            sensors = get_sensors(px, py, DIST_MAP_STATIC)
            adjust = rna_struct.forward(curr, sensors, W1, W2)
            if sensors[2] < 3.0: adjust[0]=0; adjust[1]=0
            curr += adjust * [0.08, 0.08, 0.05]
            curr = np.clip(curr, [0,0,0.02], [1,1,0.25])
            
        px = int(curr[0]*(W-1))
        py = int(curr[1]*(H-1))
        pr = int(curr[2]*W)
        
        dist = DIST_MAP_STATIC[min(py, H-1), min(px, W-1)]
        if dist > 3.0: total_score += -dist 
        else: total_score += calculate_pure_contrast(px, py, pr, TARGET_IMG)

    return -total_score / num_agents

# --- 5. MAGNETO (Ajuste Fino) ---
def magnetize_circle(cx, cy, r, img):
    best_score = calculate_pure_contrast(cx, cy, r, img)
    best_c = (cx, cy, r)
    for _ in range(20): 
        improved = False
        moves = [(0,0,1), (0,0,-1), (1,0,0), (-1,0,0), (0,1,0), (0,-1,0)]
        for dx, dy, dr in moves:
            nx, ny, nr = best_c[0]+dx, best_c[1]+dy, best_c[2]+dr
            if nr < 3: continue
            new_score = calculate_pure_contrast(nx, ny, nr, img)
            if new_score > best_score:
                best_score = new_score
                best_c = (nx, ny, nr)
                improved = True
                break 
        if not improved: break
    return best_c, best_score

# --- 6. AUXILIAR VISUAL ---
def get_check_points(cx, cy, r):
    pts_in = []
    pts_out = []
    r_in = int(r * 0.8)
    for i in range(4):
        px = int(cx + r_in * np.cos(i * np.pi/2))
        py = int(cy + r_in * np.sin(i * np.pi/2))
        pts_in.append((px, py))
    r_out = r + 2
    for i in range(8):
        theta = i * (np.pi/4)
        px = int(cx + r_out * np.cos(theta))
        py = int(cy + r_out * np.sin(theta))
        pts_out.append((px, py))
    return pts_in, pts_out

# --- 7. EXECUÇÃO SEQUENCIAL (VÍDEO) ---
def run_single_agent_video(best_bits):
    print("--- Iniciando Agente Único (Explorador) ---")
    
    # Decodifica a RNA TREINADA
    W1, W2 = rna_struct.decode_weights(best_bits)
    
    # Prepara vídeo
    output_file = 'agente_unico.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (W, H))
    
    # Imagens
    work_img = TARGET_IMG.copy() # Mapa mental (vai ficando branco)
    display_img = cv2.cvtColor(TARGET_IMG, cv2.COLOR_GRAY2BGR) # Resultado visual
    
    # Mapa inicial (dinâmico)
    _, binary_work = cv2.threshold(work_img, 127, 255, cv2.THRESH_BINARY)
    current_dist_map = cv2.distanceTransform(binary_work, cv2.DIST_L2, 5)
    
    # Posição inicial (Centro)
    curr = np.array([0.5, 0.5, 0.1])
    
    balls_found = 0
    patience = 0
    found_balls_list = []
    
    # Loop de simulação (1500 frames = 50 segundos)
    for frame_idx in range(1500):
        # 1. Sensores (Baseados no mapa ATUAL, que muda)
        px = int(np.clip(curr[0]*(W-1), 0, W-1))
        py = int(np.clip(curr[1]*(H-1), 0, H-1))
        
        sensors = get_sensors(px, py, current_dist_map)
        
        # 2. Movimento
        # IMPORTANTE: Usamos a instância rna_struct global
        adjust = rna_struct.forward(curr, sensors, W1, W2)
        
        # Velocidade variável
        speed = 0.08 if sensors[2] > 5.0 else 0.02
        curr += adjust * [speed, speed, 0.05]
        curr = np.clip(curr, [0,0,0.02], [1,1,0.25])
        
        # 3. Desenha o Frame
        frame = display_img.copy()
        
        # Agente (Vermelho com Raio-X dinâmico)
        ax, ay = int(curr[0]*(W-1)), int(curr[1]*(H-1))
        ar = int(curr[2]*W)
        
        # Desenha pontos de teste dinâmicos
        pts_in, pts_out = get_check_points(ax, ay, ar)
        for p in pts_in: cv2.circle(frame, p, 1, (255,0,0), -1)
        for p in pts_out: cv2.circle(frame, p, 1, (0,255,255), -1)
        
        # Corpo do agente
        cv2.circle(frame, (ax, ay), ar, (0, 0, 255), 2)
        cv2.circle(frame, (ax, ay), 2, (0, 0, 255), -1)
        
        cv2.putText(frame, f"Bolas: {balls_found}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        out.write(frame)
        
        # 4. Lógica de Captura
        # Se estiver perto de algo preto NO MAPA ORIGINAL
        if sensors[2] < 3.0:
            # Verifica se ainda é preto no mapa de trabalho
            if work_img[ay, ax] < 100:
                # Tenta encaixar
                final_c, final_score = magnetize_circle(ax, ay, ar, TARGET_IMG)
                
                # Se o encaixe for bom
                if final_score > 100:
                    bx, by, br = final_c
                    
                    # Validação extra: Não pode estar dentro de uma bola já achada
                    is_inside = False
                    for (fx, fy, fr) in found_balls_list:
                         dist = np.sqrt((bx-fx)**2 + (by-fy)**2)
                         # Se a distância for pequena comparada ao raio da maior, é duplicada/aninhada
                         if dist < max(br, fr) * 0.8:
                             is_inside = True
                             break
                    
                    if not is_inside:
                        balls_found += 1
                        found_balls_list.append((bx, by, br))
                        print(f"Bola {balls_found} em ({bx},{by})")
                        
                        # A. Marca Definitiva (Verde + Raio-X Fixo)
                        cv2.circle(display_img, (bx, by), br, (0, 255, 0), 2)
                        pi, po = get_check_points(bx, by, br)
                        cv2.circle(display_img, (bx, by), 2, (0, 0, 255), -1)
                        for p in pi: cv2.circle(display_img, p, 1, (255,0,0), -1)
                        for p in po: cv2.circle(display_img, p, 1, (0,255,255), -1)
                        
                        # B. Apaga do mapa mental (Pinta de branco)
                        cv2.circle(work_img, (bx, by), int(br * 1.2), 255, -1)
                        
                        # C. Atualiza gravidade
                        _, bw = cv2.threshold(work_img, 127, 255, cv2.THRESH_BINARY)
                        current_dist_map = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
                        
                        patience = 0
                        
                        # Flash visual
                        for _ in range(5):
                            flash = display_img.copy()
                            cv2.circle(flash, (bx, by), br, (0, 255, 255), 3)
                            out.write(flash)
        
        # 5. Respawn se ficar perdido
        if sensors[2] > 10.0: patience += 1
        else: patience = 0
        
        if patience > 60:
            # Teleporta para um ponto preto aleatório se houver
            black_points = np.column_stack(np.where(work_img < 50))
            if len(black_points) > 0:
                idx = np.random.randint(len(black_points))
                y, x = black_points[idx]
                curr[0] = x / W
                curr[1] = y / H
            else:
                curr = np.random.rand(3) # Random total
            curr[2] = 0.1
            patience = 0
            
        # Fim se tudo branco
        if np.min(current_dist_map) > 20.0:
            print("Limpeza concluída.")
            break

    # Congela final
    for _ in range(60): out.write(display_img)
    out.release()
    cv2.imwrite("resultado_agente_unico.png", display_img)
    print(f"Vídeo salvo: '{output_file}'")

def run():
    print("--- Treinando Cérebro ---")
    nbits = rna_struct.get_chromosome_size()
    options = {"PopulationSize": 100, "Generations": 100, "MutationFcn": 0.15, "EliteCount": 4}
    best_bits, _, fit = gago(fitness_function, nbits, options)
    print(f"Fitness: {-fit[0]:.2f}")
    
    run_single_agent_video(best_bits)

if __name__ == "__main__":
    run()