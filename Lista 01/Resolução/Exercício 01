from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Caminho relativo à pasta de execução; ajuste se necessário.
image_path = Path("Lista 01/Dados/lena2.tif")
output_path = Path("exercicio1_decimacao.png")

if not image_path.is_file():
    raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

# A conversão para L garante uma matriz 2D de cinza de 8 bits.
try:
    with Image.open(image_path) as imagem:
        lena_original = np.array(imagem.convert("L"))
except OSError as erro:
    raise OSError(f"Não foi possível ler a imagem: {image_path}") from erro

altura, largura = lena_original.shape
print(f"Original: {altura} x {largura}")

if lena_original.shape != (512, 512):
    raise ValueError(
        f"O exercício exige 512 x 512, mas a imagem tem {altura} x {largura}. "
        "Selecione a Lena gray correta; não redimensione a entrada."
    )

# Seleciona uma amostra a cada N linhas e N colunas, sem interpolação.
# copy() cria matrizes independentes e mantém a original inalterada.
lena_256 = lena_original[::2, ::2].copy()
lena_128 = lena_original[::4, ::4].copy()
lena_64 = lena_original[::8, ::8].copy()

versoes = [
    (lena_original, 1, 512),
    (lena_256, 2, 256),
    (lena_128, 4, 128),
    (lena_64, 8, 64),
]

for imagem, fator, tamanho in versoes[1:]:
    if imagem.shape != (tamanho, tamanho):
        raise ValueError(f"Dimensão incorreta na decimação de fator {fator}.")
    print(f"Decimada {fator}x: {imagem.shape[0]} x {imagem.shape[1]}")

# Painéis iguais: a resolução não determina o tamanho físico da imagem.
fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")

for ax, (imagem, fator, tamanho) in zip(axes, versoes):
    ax.imshow(
        imagem,
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        resample=False,
        aspect="equal",
    )
    titulo = "Original" if fator == 1 else f"Decimada {fator}x"
    ax.set_title(f"{titulo}\n{tamanho} × {tamanho}")
    ax.set_axis_off()

fig.savefig(output_path, dpi=200)
print(f"Figura salva em: {output_path}")
plt.show()
