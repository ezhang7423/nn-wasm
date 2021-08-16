use tract_onnx::prelude::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn get_res() -> i32 {
    match init_and_run_model() {
        Ok(v) => v.unwrap().1,
        Err(e) => panic!("{:?}", e),
    }
}

fn init_and_run_model() -> TractResult<Option<(f32, i32)>> {
    let model = tract_onnx::onnx()
        // load the model
        .model_for_path("./test.onnx")?
        // specify input type and shape
        .with_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 224, 224, 3)),
        )?
        // optimize the model
        .into_optimized()?
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?;

    println!("loaded model");
    // open image, resize it and make a Tensor out of it
    let image = image::open("./sample.png").unwrap().to_rgb8();
    // let resized =
    //     image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, x, y, c)| {
        image[(x as _, y as _)][c] as f32
    })
    .into();

    // run the model on the input
    let result = model.run(tvec!(image))?;

    // find and display the max value with its index
    // println!("{:?}", result[1]);
    let best = result[0]
        .to_array_view::<f32>()?
        .iter()
        .cloned()
        .zip(2..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    Ok(best)
}
