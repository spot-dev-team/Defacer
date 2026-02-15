
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib import colors
import matplotlib.patches as mpatches

# --- CONFIGURAÇÃO ---
# Substitui pelo caminho do teu ficheiro .nii.gz local
CAMINHO_ARQUIVO = r"C:\Tese\ADNI_Lite_Cluster\002_S_0413__2006-11-15_14_23_260__I30119\training_masks\mask_4_classes.nii.gz" 
# --- LÓGICA DE LIMPEZA (IGUAL AO TREINO) ---
def limpar_e_remapear(mask_data):
    """
    Remove o lixo (Label 1) e reordena os órgãos.
    Entrada Slicer: 0=Fundo, 1=Lixo, 2=Orelhas, 3=Boca, 4=Nariz, 5=Olhos
    Saída Visual:   0=Fundo, 1=Orelhas, 2=Boca, 3=Nariz, 4=Olhos
    """
    new_mask = np.zeros_like(mask_data)
    new_mask[mask_data == 2] = 1 # Orelhas (Vermelho)
    new_mask[mask_data == 3] = 2 # Boca (Verde)
    new_mask[mask_data == 4] = 3 # Nariz (Azul)
    new_mask[mask_data == 5] = 4 # Olhos (Amarelo)
    return new_mask

# --- CARREGAR DADOS ---
print(f"A carregar: {CAMINHO_ARQUIVO}...")
try:
    nii = nib.load(CAMINHO_ARQUIVO)
    dados_originais = nii.get_fdata()
    dados_limpos = limpar_e_remapear(dados_originais)
except Exception as e:
    print(f"Erro ao abrir o ficheiro: {e}")
    exit()

# Configuração de Cores
# 0: Transparente (Fundo), 1: Vermelho, 2: Verde, 3: Azul, 4: Amarelo
cmap = colors.ListedColormap(['black', 'red', 'lime', 'blue', 'yellow'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

# --- VISUALIZADOR ---
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.25) # Espaço para o slider

# Definir vista inicial (Eixo Coronal - Y é geralmente o melhor para ver a cara)
eixo_atual = 1 # 0=Sagital, 1=Coronal, 2=Axial
max_fatias = dados_limpos.shape[eixo_atual]
fatia_inicial = max_fatias // 2

# Função auxiliar para ir buscar a fatia correta dependendo do eixo
def get_slice(idx, axis):
    if axis == 0: return np.rot90(dados_limpos[idx, :, :])
    if axis == 1: return np.rot90(dados_limpos[:, idx, :])
    if axis == 2: return np.rot90(dados_limpos[:, :, idx])

# Mostrar imagem inicial
img_plot = ax.imshow(get_slice(fatia_inicial, eixo_atual), cmap=cmap, norm=norm)
ax.set_title(f"Visualizador de Segmentação - Fatia {fatia_inicial}")
ax.axis('off')

# Criar Legenda
legendas = [
    mpatches.Patch(color='black', label='Fundo / Lixo Removido'),
    mpatches.Patch(color='red',   label='Orelhas'),
    mpatches.Patch(color='lime',  label='Boca'),
    mpatches.Patch(color='blue',  label='Nariz'),
    mpatches.Patch(color='yellow',label='Olhos')
]
ax.legend(handles=legendas, loc='upper right', bbox_to_anchor=(1.3, 1))

# --- SLIDER (A BARRA DESLIZANTE) ---
ax_slider = plt.axes([0.2, 0.1, 0.60, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(
    ax=ax_slider,
    label='Fatia',
    valmin=0,
    valmax=max_fatias - 1,
    valinit=fatia_inicial,
    valstep=1
)

# --- BOTÕES PARA MUDAR DE EIXO (OPCIONAL) ---
rax = plt.axes([0.02, 0.4, 0.12, 0.15], facecolor='lightgray')
radio = RadioButtons(rax, ('Sagital (X)', 'Coronal (Y)', 'Axial (Z)'), active=1)

# Função de Atualização
def update(val):
    idx = int(slider.val)
    # Atualiza a imagem
    img_plot.set_data(get_slice(idx, eixo_atual))
    ax.set_title(f"Visualizador - Fatia {idx}")
    fig.canvas.draw_idle()

def change_axis(label):
    global eixo_atual, max_fatias
    if label == 'Sagital (X)': eixo_atual = 0
    elif label == 'Coronal (Y)': eixo_atual = 1
    elif label == 'Axial (Z)': eixo_atual = 2
    
    # Atualizar limites do slider porque as dimensões mudam
    max_fatias = dados_limpos.shape[eixo_atual]
    slider.valmax = max_fatias - 1
    slider.val = max_fatias // 2
    slider.ax.set_xlim(0, max_fatias - 1)
    
    update(slider.val)

slider.on_changed(update)
radio.on_clicked(change_axis)

plt.show()