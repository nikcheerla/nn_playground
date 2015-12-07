var nn = require('synaptic');



var myLSTM = new nn.Architect.LSTM(1,8,1);

var learningRate = .3;
var p = 0.5;

//ballpark estimates a cumulative probability
for (var i = 0; i < 20000; i++)
{
	var dp = Math.random();
	p = (p + dp)*(dp)
    // 0,0 => 0
    myLSTM.activate([dp]);
    myLSTM.propagate(learningRate, [p]);
}

var dp = Math.random();
p = (p + dp)*(dp)
console.log(p);
console.log(myLSTM.activate([dp])); 
