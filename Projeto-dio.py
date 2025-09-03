import numpy as np
from PIL import Image

def converter_para_cinza_numpy(imagem):
    """
    Converte uma imagem colorida para tons de cinza usando NumPy.
    """
    # Carregar a imagem usando PIL
    img = Image.open(imagem).convert('RGB')
    img_array = np.array(img)

    # Fórmula da luminância (valores entre 0 e 255)
    img_gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]

    return img_gray.astype(np.uint8)


def binarizar(img_gray, limiar=128):
    """
    Converte uma imagem em tons de cinza para binária (0 e 255) usando um limiar fixo.
    """
    img_bin = np.where(img_gray >= limiar, 255, 0)
    return img_bin.astype(np.uint8)


def otsu_threshold(img_gray):
    """
    Calcula o limiar ótimo usando o método de Otsu.
    """
    hist, _ = np.histogram(img_gray.ravel(), bins=256, range=(0, 256))
    total = img_gray.size
    sum_total = np.dot(np.arange(256), hist)

    sumB, wB, max_var, best_t = 0.0, 0.0, 0.0, 0
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        between_var = wB * wF * (mB - mF) ** 2
        if between_var > max_var:
            max_var = between_var
            best_t = t
    return best_t


# ------------------------------
# Exemplo de uso
# ------------------------------
imagem = "lena.jpg"   # coloque o nome/caminho da sua imagem aqui

# 1) Converter para cinza
img_cinza = converter_para_cinza_numpy(imagem)
Image.fromarray(img_cinza).save("lena_cinza.jpg")

# 2) Binarização com limiar fixo
img_binaria_fixa = binarizar(img_cinza, limiar=128)
Image.fromarray(img_binaria_fixa).save("lena_binaria_fixa.jpg")

# 3) Binarização com Otsu
limiar_otsu = otsu_threshold(img_cinza)
img_binaria_otsu = binarizar(img_cinza, limiar=limiar_otsu)
Image.fromarray(img_binaria_otsu).save("lena_binaria_otsu.jpg")

print(f"✔ Imagem em tons de cinza salva como lena_cinza.jpg")
print(f"✔ Imagem binária (fixa, T=128) salva como lena_binaria_fixa.jpg")
print(f"✔ Imagem binária (Otsu, T={limiar_otsu}) salva como lena_binaria_otsu.jpg")

