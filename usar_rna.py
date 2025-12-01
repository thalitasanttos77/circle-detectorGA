import numpy as np
import cv2
import sys
import os

# Configuração de Importação
sys.path.append(os.path.abspath("Python"))
try:
    from gapy.utilities import bits2bytes
except ImportError:
    print("ERRO: Verifique a pasta 'Python/gapy'.")
    sys.exit(1)

# --- 1. ESTRUTURA DA RNA (Idêntica ao Treino) ---
class MaskRNA:
    def __init__(self):
        # 4 entradas: GradX, GradY, Dist, Raio
        self.input_size = 4 
        self.hidden_size = 6
        self.output_size = 3
        self.w1_shape = (self.input_size + 1, self.hidden_size)
        self.w2_shape = (self.hidden_size + 1, self.output_size)
    
    def decode_weights(self, bits):
        bytes_data = bits2bytes(bits, dtype='int8')
        # Divisão por 10.0 (Mesma força do treino)
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

# --- 2. SENSORES E AUXILIARES ---
def get_sensors(px, py, dist_map):
    h, w = dist_map.shape
    px = int(np.clip(px, 2, w-3))
    py = int(np.clip(py, 2, h-3))
    gx = (dist_map[py, px+2] - dist_map[py, px-2]) / 2.0
    gy = (dist_map[py+2, px] - dist_map[py-2, px]) / 2.0
    dist = dist_map[py, px]
    return gx, gy, dist

def calculate_contrast(cx, cy, r, img):
    h, w = img.shape
    if cx < 0 or cx >= w or cy < 0 or cy >= h: return -200.0
    if r < 4: return -200.0
    try:
        mask_in = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_in, (cx, cy), int(r * 0.7), 255, -1)
        mean_in = cv2.mean(img, mask=mask_in)[0]

        mask_out = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_out, (cx, cy), int(r + 3), 255, -1)
        cv2.circle(mask_out, (cx, cy), int(r), 0, -1)
        mean_out = cv2.mean(img, mask=mask_out)[0]
        return mean_out - mean_in
    except: return -200.0

def magnetize_circle(cx, cy, r, img):
    """Refinamento final (Hill Climbing)"""
    best_score = calculate_contrast(cx, cy, r, img)
    best_c = (cx, cy, r)
    for _ in range(20):
        improved = False
        moves = [(0,0,1), (0,0,-1), (1,0,0), (-1,0,0), (0,1,0), (0,-1,0)]
        for dx, dy, dr in moves:
            nx, ny, nr = best_c[0]+dx, best_c[1]+dy, best_c[2]+dr
            if nr < 3: continue
            new_score = calculate_contrast(nx, ny, nr, img)
            if new_score > best_score:
                best_score = new_score
                best_c = (nx, ny, nr)
                improved = True
                break 
        if not improved: break
    return best_c, best_score

def remove_overlaps(candidates):
    """Limpeza rigorosa de duplicatas"""
    candidates.sort(key=lambda x: x[3], reverse=True) # Ordena por score
    unique = []
    for c in candidates:
        cx, cy, cr, cscore = c
        is_parasite = False
        for ux, uy, ur, uscore in unique:
            dist = np.sqrt((cx-ux)**2 + (cy-uy)**2)
            # Se está dentro ou sobrepõe muito (mesma lógica do treino)
            if dist + cr <= ur + 2 or dist < max(cr, ur):
                is_parasite = True
                break
        if not is_parasite:
            unique.append(c)
    return unique

# --- 3. EXECUÇÃO PRINCIPAL ---
def aplicar_modelo(imagem_arquivo, modelo_arquivo="melhor_rna.npy"):
    if not os.path.exists(modelo_arquivo):
        print("ERRO: 'melhor_rna.npy' não encontrado.")
        return
    
    if not os.path.exists(imagem_arquivo):
        print(f"ERRO: Imagem '{imagem_arquivo}' não encontrada.")
        return

    # A. Carrega Imagem e Prepara Mapas
    img = cv2.imread(imagem_arquivo, cv2.IMREAD_GRAYSCALE)
    if img is None: return
    H, W = img.shape
    print(f"Analisando imagem existente '{imagem_arquivo}' ({W}x{H})...")

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # B. Carrega Cérebro
    bits = np.load(modelo_arquivo)
    rna = MaskRNA()
    W1, W2 = rna.decode_weights(bits)

    # C. Gera Grade de Agentes (Varredura Completa)
    grid_size = 20
    grid_x, grid_y = np.meshgrid(np.linspace(0.05, 0.95, grid_size), np.linspace(0.05, 0.95, grid_size))
    agents = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    radii = np.random.rand(len(agents), 1) * 0.1 + 0.05
    agents = np.hstack((agents, radii))

    candidates = []
    
    # D. Loop da IA
    print(f"Lançando {len(agents)} agentes...")
    for state in agents:
        curr = state.copy()
        
        # Caminhada
        for _ in range(50):
            px = int(np.clip(curr[0]*(W-1), 0, W-1))
            py = int(np.clip(curr[1]*(H-1), 0, H-1))
            
            sensors = get_sensors(px, py, dist_map)
            adjust = rna.forward(curr, sensors, W1, W2)
            
            if sensors[2] < 3.0: adjust[0]=0; adjust[1]=0
            
            curr += adjust * [0.08, 0.08, 0.05]
            curr = np.clip(curr, [0,0,0.02], [1,1,0.25])

        # E. Refinamento
        x, y = int(curr[0]*(W-1)), int(curr[1]*(H-1))
        r = int(curr[2]*W)
        
        dist = dist_map[min(y, H-1), min(x, W-1)]
        if dist < 5.0:
            final_c, final_score = magnetize_circle(x, y, r, img)
            if final_score > 100:
                candidates.append((*final_c, final_score))

    # F. Limpeza e Resultado
    unique_balls = remove_overlaps(candidates)
    
    # Gera imagem de saída
    res_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    print(f"Bolas Detectadas: {len(unique_balls)}")

    for (x, y, r, score) in unique_balls:
        cv2.circle(res_img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(res_img, (x, y), 2, (0, 0, 255), -1)

    output_file = "resultado_usar_rna.png"
    cv2.imwrite(output_file, res_img)
    print(f"Resultado salvo em '{output_file}'")

if __name__ == "__main__":
    aplicar_modelo("ImagemDemo.png")