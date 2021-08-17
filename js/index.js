import("../pkg/index.js")
  .catch(console.error)
  .then(async (module) => {
    window.nn = module;

    console.log("initializing network");
    let t0 = performance.now();
    let network = await module.Network.new();
    let t1 = performance.now();
    console.log(`loading network took ${(t1 - t0) / 1000} seconds`);
    console.log("getting prediction");

    console.log("pred:", network.run());
    let t2 = performance.now();
    console.log(`inference took ${(t2 - t1) / 1000} seconds`);
  });
