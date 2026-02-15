import pandas as pd
import re
import os

# --- CONFIGURAÇÃO ---
CSV_FILE = 'training_log_final.csv'       # Nome do teu CSV
LOG_FILE = './logs/treino2_173519.log'         # Nome do teu Log (texto)
# --------------------

def extract_info():
    results = {}
    
    # 1. PROCESSAR CSV (Melhor Época e Métricas de Treino)
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            # Encontrar a linha com o melhor Val Dice Score
            best_idx = df['val_dice_score'].idxmax()
            best_row = df.iloc[best_idx]
            
            results['Best Epoch'] = int(best_row['epoch']) + 1 # +1 porque começa em 0
            results['Train Dice'] = float(best_row['dice_score'])
            results['Val Dice'] = float(best_row['val_dice_score'])
            results['Train Loss'] = float(best_row['loss'])
            results['Val Loss'] = float(best_row['val_loss'])
            
            # Verificar Stop Reason pelo CSV
            if len(df) == 60:
                results['Stop Reason'] = "Max Epochs (60)"
            else:
                results['Stop Reason'] = f"Early Stopping / Crash (Ep {len(df)})"
                
        except Exception as e:
            print(f"Erro ao ler CSV: {e}")
    else:
        print(f"⚠️ Aviso: CSV '{CSV_FILE}' não encontrado.")

    # 2. PROCESSAR LOG (Resultados Finais de Teste)
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
            
            # Procurar Dice Score do ADNI (Dataset 1)
            # Regex procura por "ADNI Test Dice: 0.1234"
            match_adni = re.search(r"ADNI Test Dice:\s*([\d\.]+)", log_content)
            if match_adni:
                results['Dice Test Score D1 (ADNI)'] = float(match_adni.group(1))
            else:
                results['Dice Test Score D1 (ADNI)'] = "N/A (Não encontrado)"

            # Procurar Dice Score do IXI (Dataset 2)
            match_ixi = re.search(r"IXI External Test Dice:\s*([\d\.]+)", log_content)
            if match_ixi:
                results['Dice Test Score D2 (IXI)'] = float(match_ixi.group(1))
            else:
                results['Dice Test Score D2 (IXI)'] = "N/A (Não usado/Crash)"

            # Tentar confirmar o Stop Reason se houver mensagem explícita
            if "early stopping" in log_content.lower():
                 results['Stop Reason'] += " [Confirmado no Log]"

        except Exception as e:
            print(f"Erro ao ler Log: {e}")
    else:
        print(f"⚠️ Aviso: Log '{LOG_FILE}' não encontrado.")

    # 3. APRESENTAR RESULTADOS
    print("\n" + "="*40)
    print(f"   RELATÓRIO AUTOMÁTICO PARA EXCEL")
    print("="*40)
    
    order = [
        'Best Epoch', 'Stop Reason', 
        'Train Dice', 'Val Dice', 
        'Train Loss', 'Val Loss',
        'Dice Test Score D1 (ADNI)', 'Dice Test Score D2 (IXI)'
    ]
    
    for key in order:
        val = results.get(key, '---')
        if isinstance(val, float):
            print(f"{key:.<30} {val:.4f}")
        else:
            print(f"{key:.<30} {val}")
    print("="*40)

if __name__ == "__main__":
    extract_info()