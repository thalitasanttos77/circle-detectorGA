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
DIST_MAP = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

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

# --- 4. SCORE PURO (Contraste Mássico) ---
def calculate_pure_contrast(cx, cy, r, img):
    h, w = img.shape
    if cx < 0 or cx >= w or cy < 0 or cy >= h: return -200.0
    if r < 4: return -200.0

    try:
        # Miolo (0.8 do raio) - Deve ser PRETO
        mask_in = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_in, (cx, cy), int(r * 0.8), 255, -1)
        mean_in = cv2.mean(img, mask=mask_in)[0]

        # Borda (Anel entre r e r+3) - Deve ser BRANCA
        mask_out = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_out, (cx, cy), int(r + 3), 255, -1)
        cv2.circle(mask_out, (cx, cy), int(r), 0, -1)
        mean_out = cv2.mean(img, mask=mask_out)[0]

        # Quanto maior a diferença, melhor
        contrast = mean_out - mean_in
        
        # Leve penalidade para não explodir o raio
        penalty = r * 0.05 
        
        return contrast - penalty
    except:
        return -200.0

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
            sensors = get_sensors(px, py, DIST_MAP)
            adjust = rna_struct.forward(curr, sensors, W1, W2)
            if sensors[2] < 3.0: adjust[0]=0; adjust[1]=0
            curr += adjust * [0.08, 0.08, 0.05]
            curr = np.clip(curr, [0,0,0.02], [1,1,0.25])
            
        px = int(curr[0]*(W-1))
        py = int(curr[1]*(H-1))
        pr = int(curr[2]*W)
        dist = DIST_MAP[min(py, H-1), min(px, W-1)]
        if dist > 3.0: total_score += -dist 
        else: total_score += calculate_pure_contrast(px, py, pr, TARGET_IMG)

    return -total_score / num_agents

# --- 5. MAGNETO (Ajuste Fino) ---
def magnetize_circle(cx, cy, r, img):
    best_score = calculate_pure_contrast(cx, cy, r, img)
    best_c = (cx, cy, r)
    for _ in range(25): 
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

# --- 6. LIMPEZA (NMS) ---
def remove_overlaps_quality(candidates):
    candidates.sort(key=lambda x: x[3], reverse=True)
    unique = []
    for c in candidates:
        cx, cy, cr, cscore = c
        is_parasite = False
        for ux, uy, ur, uscore in unique:
            dist = np.sqrt((cx-ux)**2 + (cy-uy)**2)
            if dist < (cr + ur) * 0.5:
                is_parasite = True; break
        if not is_parasite: unique.append(c)
    return unique

# --- 7. EXECUÇÃO ---
def run():
    print("--- Busca Final: Score Puro + Debug Visual ---")
    
    nbits = rna_struct.get_chromosome_size()
    options = {
        "PopulationSize": 100, 
        "Generations": 120,
        "MutationFcn": 0.15,
        "EliteCount": 4
    }

    print("1. Treinando...")
    best_bits, _, fit_vals = gago(fitness_function, nbits, options)
    print(f"   Fitness: {-fit_vals[0]:.2f}")

    np.save("melhor_rna.npy", best_bits)
    visualize(best_bits)

def visualize(bits):
    W1, W2 = rna_struct.decode_weights(bits)
    demo = cv2.cvtColor(TARGET_IMG, cv2.COLOR_GRAY2BGR)
    
    # Grade de Varredura
    grid_size = 20
    grid_x, grid_y = np.meshgrid(np.linspace(0.05, 0.95, grid_size), np.linspace(0.05, 0.95, grid_size))
    agents = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    radii = np.random.rand(len(agents), 1) * 0.1 + 0.05
    agents = np.hstack((agents, radii))
    
    candidates = []
    print(f"2. Varrendo com {len(agents)} agentes...")
    
    for state in agents:
        curr = state.copy()
        for _ in range(50):
            px = int(np.clip(curr[0]*(W-1), 0, W-1))
            py = int(np.clip(curr[1]*(H-1), 0, H-1))
            sensors = get_sensors(px, py, DIST_MAP)
            adjust = rna_struct.forward(curr, sensors, W1, W2)
            if sensors[2] < 3.0: adjust[0]=0; adjust[1]=0
            curr += adjust * [0.08, 0.08, 0.05]
            curr = np.clip(curr, [0,0,0.02], [1,1,0.25])

        x, y = int(curr[0]*(W-1)), int(curr[1]*(H-1))
        r = int(curr[2]*W)
        
        dist = DIST_MAP[min(y, H-1), min(x, W-1)]
        if dist < 5.0:
            final_c, final_score = magnetize_circle(x, y, r, TARGET_IMG)
            if final_score > 120:
                candidates.append((*final_c, final_score))

    unique_balls = remove_overlaps_quality(candidates)
    
    print(f"3. Resultado Final: {len(unique_balls)} bolas.")

    for (x, y, r, score) in unique_balls:
        # Círculo Verde
        cv2.circle(demo, (x, y), r, (0, 255, 0), 2)
        
        # --- DESENHO DOS PONTOS DE VALIDAÇÃO ---
        
        # 1. Centro (Vermelho)
        cv2.circle(demo, (x, y), 2, (0, 0, 255), -1)
        
        # 2. Amostras Internas (Azul - Região do Miolo 0.8)
        # 8 pontos
        r_in = int(r * 0.8)
        for i in range(8):
            theta = (i * 2 * np.pi) / 8
            px = int(x + r_in * np.cos(theta))
            py = int(y + r_in * np.sin(theta))
            cv2.circle(demo, (px, py), 2, (255, 0, 0), -1) # Azul

        # 3. Amostras Externas (Amarelo - Região da Borda ~r+2)
        r_out = r + 2
        for i in range(8):
            theta = (i * 2 * np.pi) / 8
            px = int(x + r_out * np.cos(theta))
            py = int(y + r_out * np.sin(theta))
            cv2.circle(demo, (px, py), 2, (0, 255, 255), -1) # Amarelo (Cyan em BGR)

    cv2.imwrite("resultado_final.png", demo)
    print("Salvo em 'resultado_final.png'")

if __name__ == "__main__":
    run()