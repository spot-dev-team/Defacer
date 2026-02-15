import nibabel as nib
import os
import numpy as np
import pandas as pd

# --- CONFIGURAÇÃO ---
IXI_FOLDER = "D:\\Tese - BDs\\IXI-T1"  # Muda isto

def inspect_nifti_header(folder):
    report = []
    
    files = [f for f in os.listdir(folder) if f.endswith('.nii.gz') or f.endswith('.nii')]
    print(f"A analisar {len(files)} ficheiros...")
    
    for f in files:
        path = os.path.join(folder, f)
        try:
            img = nib.load(path)
            header = img.header
            
            # Dimensões (Ex: 256, 256, 150)
            dims = header.get_data_shape()
            
            # Tamanho do Voxel (Ex: 0.93, 0.93, 1.2)
            zooms = header.get_zooms()
            
            # Orientação (Aproximada)
            aff = img.affine
            orientation = nib.aff2axcodes(aff)
            
            # Garante que olhamos apenas para as 3 primeiras dimensões (X, Y, Z)
            # (Alguns Niftis têm uma 4ª dimensão temporal que não interessa)
            x_res, y_res, z_res = zooms[:3]
            
            # CRITÉRIO DE ACEITAÇÃO:
            # ADNI típico é 1.0 x 1.0 x 1.2 mm.
            # Vamos aceitar uma margem entre 0.8 e 1.3 mm em todos os eixos.
            is_compatible = (0.8 <= x_res <= 1.3) and \
                            (0.8 <= y_res <= 1.3) and \
                            (0.8 <= z_res <= 1.3)

            status = "✅ SIM"
            if not is_compatible:
                status = f"❌ NÃO (Voxel: {x_res:.2f}x{y_res:.2f}x{z_res:.2f})"

            report.append({
                "Filename": f,
                "Dimensions": dims,
                "Voxel Size (mm)": [round(z, 2) for z in zooms[:3]],
                "Orientation": orientation,
                "Compatible": status
            })
        except Exception as e:
            print(f"Erro em {f}: {e}")

    # Criar Tabela Bonita
    df = pd.DataFrame(report)
    return df

if __name__ == "__main__":
    if os.path.exists(IXI_FOLDER):
        df = inspect_nifti_header(IXI_FOLDER)
        print("\n--- RELATÓRIO DE COMPATIBILIDADE IXI ---")
        print(df.head(20).to_string())
        
        # Guardar CSV para a tese se quiseres
        df.to_csv("ixi_dataset_specs.csv", index=False)
        print("\nRelatório salvo em 'ixi_dataset_specs.csv'")
    else:
        print("Pasta não encontrada.")