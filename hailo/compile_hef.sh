#!/usr/bin/env bash
# ONNX -> Hailo-8L HEF. Run inside WSL Ubuntu 22.04 (or any x86 Ubuntu 22.04), from this folder.
# Put here first:
#   best.onnx, calib.zip                       <- from train/colab_train.ipynb
#   hailo_dataflow_compiler-*.whl              <- hailo.ai Developer Zone (free login), Linux, Python 3.10
#   hailo_model_zoo-*.whl                      <- same page
# Pick the DFC / Model Zoo release that matches the Pi's HailoRT (`hailortcli --version` on the Pi),
# otherwise the Pi refuses the HEF.
set -euo pipefail
cd "$(dirname "$0")"
ls best.onnx calib.zip hailo_dataflow_compiler-*.whl hailo_model_zoo-*.whl >/dev/null

if [ ! -d ~/hailo_venv ]; then
  sudo apt-get update
  sudo apt-get install -y python3.10-venv python3.10-dev build-essential graphviz graphviz-dev unzip
  python3.10 -m venv ~/hailo_venv
  ~/hailo_venv/bin/pip install --upgrade pip
  ~/hailo_venv/bin/pip install hailo_dataflow_compiler-*.whl hailo_model_zoo-*.whl
fi
. ~/hailo_venv/bin/activate

rm -rf calib && unzip -q calib.zip -d calib
# No NVIDIA GPU in WSL here -> DFC uses a lower optimization level (slightly lower int8 accuracy). Takes ~30-90 min.
hailomz compile yolov8n --ckpt=best.onnx --hw-arch hailo8l --calib-path calib/ --classes 1
mv yolov8n.hef drone_yolov8n.hef
echo "done: copy drone_yolov8n.hef to the Pi's interceptor/ folder"
