extern crate wasm_tract;


#[cfg(not(target_arch = "wasm32"))]
#[tokio::main]
async fn main() {
    println!("Testing output: {}", wasm_tract::get_res())
}
