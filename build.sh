#!/bin/sh
cd "$(dirname "$0")"

python3 gen_model.py # generate .pb
python -m tf2onnx.convert --saved-model . --output test.onnx # generate onnx
cp test.onnx static
cargo run
wasm-pack build --dev # generate wasm
