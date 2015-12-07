var nn = require('synaptic');


// create the network
var inputLayer = new nn.Layer(2);
var hiddenLayer = new nn.Layer(3);
var outputLayer = new nn.Layer(1);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

var myNetwork = new nn.Network({
    input: inputLayer,
    hidden: [hiddenLayer],
    output: outputLayer
});

// train the network
var learningRate = .3;
for (var i = 0; i < 20000; i++)
{
    // 0,0 => 0
    myNetwork.activate([0,0]);
    myNetwork.propagate(learningRate, [0]);

    // 0,1 => 1
    myNetwork.activate([0,1]);
    myNetwork.propagate(learningRate, [1]);

    // 1,0 => 1
    myNetwork.activate([1,0]);
    myNetwork.propagate(learningRate, [1]);

    // 1,1 => 0
    myNetwork.activate([1,1]);
    myNetwork.propagate(learningRate, [0]);
}


// test the network
console.log(myNetwork.activate([0,0])); // [0.015020775950893527]
console.log(myNetwork.activate([0,1])); // [0.9815816381088985]
console.log(myNetwork.activate([1,0])); // [0.9871822457132193]
console.log(myNetwork.activate([1,1])); // [0.012950087641929467]