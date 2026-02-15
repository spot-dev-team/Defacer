# mover exames do _REJECTED de volta para a pasta original

import os
import shutil

BASE_DIR = r"D:\Tese_BDs\Defacer\ADNI_Structured"
REJECT_DIR = os.path.join(BASE_DIR, "_REJECTED")

def restore_rejected():
    if not os.path.exists(REJECT_DIR):
        print("Pasta _REJECTED não existe. Nada a fazer.")
        return

    print("A restaurar exames rejeitados...")
    count = 0
    
    for folder in os.listdir(REJECT_DIR):
        src = os.path.join(REJECT_DIR, folder)
        dst = os.path.join(BASE_DIR, folder)
        
        if os.path.isdir(src):
            if not os.path.exists(dst):
                shutil.move(src, dst)
                print(f" -> Restaurado: {folder}")
                count += 1
            else:
                print(f" -> [AVISO] Já existe na origem: {folder}")

    # Apagar pasta vazia
    if not os.listdir(REJECT_DIR):
        os.rmdir(REJECT_DIR)
        
    print(f"Total restaurado: {count}")

if __name__ == "__main__":
    restore_rejected()