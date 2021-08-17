use image::io::Reader as ImageReader;
use std::io::Cursor;
use tract_onnx::prelude::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub async fn get_res() -> i32 {
    match init_and_run_model().await {
        Ok(v) => v.unwrap().1,
        Err(e) => panic!("{:?}", e),
    }
}

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

async fn init_and_run_model() -> TractResult<Option<(f32, i32)>> {
    console_error_panic_hook::set_once();
    println!("sad boi af");

    // let mut data = Vec::new();
    let body = reqwest::get("http://localhost:5500/static/test.onnx")
        .await?
        .bytes()
        .await?;
    println!("sad bfe af");
    // let mut reader = res.into_reader();
    // let data = [].to_vec();
    let mut reader = Cursor::new(body);
    let model = tract_onnx::onnx()
        // load the model
        .model_for_read(&mut reader)?
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
    unsafe {
        log("loaded the  model");
    }
    // open image, resize it and make a Tensor out of itW
    let img_bytes = reqwest::get("http://localhost:5500/static/sample.png")
        .await?
        .bytes()
        .await?;
    let image = ImageReader::with_format(Cursor::new(img_bytes), image::ImageFormat::Png)
        .decode()?
        .to_rgb8();
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

#[wasm_bindgen]
pub struct bruh {
    pub but_nuh: i32,
    yuh: Tensor,
}

#[wasm_bindgen]
impl bruh {
    pub fn new() -> Self {
        let tensor = match Tensor::zero::<i32>(&[1, 224, 224, 3]) {
            Ok(v) => v,
            Err(e) => panic!("{:?}", e),
        };
        bruh {
            but_nuh: 0,
            yuh: tensor,
        }
    }
    pub fn excite(&self) {
        unsafe {
            log(&format!("{:?}", self.yuh));
        }
    }
}
impl bruh {
    fn prive() {
        unsafe { log("hha") }
    }
}
