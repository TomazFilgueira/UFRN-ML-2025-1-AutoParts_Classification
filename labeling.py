import cv2
import os

# Diretório base onde os dados estão salvos
BASE_DIR = "dataset_iracing"

# Lista de rótulos possíveis
LABELS = [
    "reta",
    "freada",
    "curva_apex",
    "saida_curva",
    "carro_a_frente",
    "carro_atras",
    "disputa_lado_a_lado",
    "pista_livre",
    "outros"
]

# Caminho para as imagens não rotuladas
UNLABELED_DIR = os.path.join(BASE_DIR, "screenshots")

# Certifique-se de que o diretório existe
if not os.path.exists(UNLABELED_DIR):
    print(f"Diretório {UNLABELED_DIR} não encontrado.")
else:
    images = [f for f in os.listdir(UNLABELED_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_name in images:
        img_path = os.path.join(UNLABELED_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Não foi possível abrir {img_name}.")
            continue
        cv2.imshow("Imagem para rotular", img)
        print("Escolha o rótulo para esta imagem:")
        for i, label in enumerate(LABELS):
            print(f"{i} - {label}")
        choice = input("Digite o número correspondente: ")

        try:
            label = LABELS[int(choice)]
        except (ValueError, IndexError):
            print("Opção inválida. Imagem descartada.")
            cv2.destroyAllWindows()
            continue

        # Move a imagem para a pasta do rótulo escolhido
        dest_dir = os.path.join(BASE_DIR, label)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, img_name)
        cv2.destroyAllWindows()
        os.rename(img_path, dest_path)
        print(f"Imagem movida para: {dest_path}")