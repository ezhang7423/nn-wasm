use image::io::Reader as ImageReader;
use std::io::Cursor;
use tract_onnx::prelude::*;
use wasm_bindgen::prelude::*;

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

async fn init_model() -> TractResult<(Model, Tensor)> {
    // let mut data = Vec::new();
    let body = reqwest::get("http://localhost:5500/static/test.onnx")
        .await?
        .bytes()
        .await?;
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

    log("loaded the  model");

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

    log("loaded the  image");

    Ok((model, image))
}

#[wasm_bindgen]
pub struct Network {
    model: Model,
    image: Tensor,
}

#[wasm_bindgen]
impl Network {
    pub async fn new() -> Self {
        console_error_panic_hook::set_once();
        println!("Started init");
        let vals = match init_model().await {
            Ok(v) => v,
            Err(e) => panic!("{:?}", e),
        };
        Network {
            model: vals.0,
            image: vals.1,
        }
    }
    pub fn run(&self) -> i32 {
        // run the model on the input
        log("a");
        let run_input = tvec!(self.image.clone());
        log("b");
        let result = self.model.run(run_input).unwrap();
        // find and display the max value with its index
        // println!("{:?}", result[1]);
        log("c");
        let best = result[0]
            .to_array_view::<f32>()
            .unwrap()
            .iter()
            .cloned()
            .zip(2..)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap();
        best.1
    }
}
