#!/bin/sh
cd "$(dirname "$0")"

python3 gen_model.py # generate .pb
python -m tf2onnx.convert --saved-model . --output model.onnx # generate onnx
cargo run
wasm-pack build --dev # generate wasm
