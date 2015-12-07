var nn = require('synaptic');

function isPrime(n) {
    if (isNaN(n) || !isFinite(n) || n%1 || n<2) return false; 
    var m=Math.sqrt(n);
    for (var i=2;i<=m;i++) if (n%i==0) return false;
    return true;
}

// create the network
var inputLayer = new nn.Layer(1);
var hiddenLayer = new nn.Layer(40);
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
for (var i = 0; i < 2000; i++)
{
    console.log(i);
    for (var k = 0; k < 1000; k++)
    {
        myNetwork.activate([k/2000]);
        if(isPrime(k)) {
            myNetwork.propagate(learningRate, [k/1000]);
        }
        else {
            myNetwork.propagate(learningRate, [k/1000]);
        }
    }
}


for (var k = 0; k < 1000; k++)
{
    console.log(k);
    console.log("Is prime: " + isPrime(k));
    console.log("Activated: " + myNetwork.activate([k/2000]));
}