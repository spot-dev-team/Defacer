#!/bin/bash

# Argumentos recebidos do Python (Caminhos estilo Linux/WSL)
# $1 = Caminho do ficheiro NIfTI de entrada (Raw)
# $2 = Pasta de Output para este exame
INPUT_FILE="$1"
OUTPUT_DIR="$2"

# 1. Configurar FreeSurfer (Ajusta se o caminho for diferente no teu WSL)
export FREESURFER_HOME=/usr/local/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh > /dev/null 2>&1

# Criar output dir se não existir
mkdir -p "$OUTPUT_DIR"

# 2. Executar MiDeFace
# Gera: face_mask.nii.gz (Mascara) e a pasta qa/samseg (Estruturas)
# Nota: --odir define onde ficam os ficheiros auxiliares (QA, Samseg)
mideface --i "$INPUT_FILE" \
         --o "$OUTPUT_DIR/mideface_anon_temp.nii.gz" \
         --facemask "$OUTPUT_DIR/face_mask.nii.gz" \
         --odir "$OUTPUT_DIR/qa" \
         --threads 4

# 3. Converter Segmentação Interna (Olhos) para NIfTI
# O Samseg gera 'seg.mgz'. Convertemos para 'eyes_raw.nii.gz'
SAMSEG_FILE="$OUTPUT_DIR/qa/samseg/seg.mgz"
if [ -f "$SAMSEG_FILE" ]; then
    mri_convert "$SAMSEG_FILE" "$OUTPUT_DIR/eyes_raw.nii.gz" > /dev/null 2>&1
else
    echo "[ERRO BASH] Ficheiro Samseg não encontrado!"
    exit 1
fi