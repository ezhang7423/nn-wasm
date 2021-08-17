import("../pkg/index.js")
  .catch(console.error)
  .then((module) => {
    window.nn = module;
    console.log("getting prediction");
    module.get_res().then((res) => {
      console.log("prediction is", res);
    });
  });
