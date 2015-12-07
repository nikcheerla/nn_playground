var nn = require('synaptic');
var fs = require('fs')
var _ = require('underscore')

txt = fs.readFileSync('datasets/shakespeare.txt');

var chars = _.uniq(txt);

console.log("Text Length:" + txt.length);
console.log("Chars Length:" + chars.length);

charArr = [];
for (var i = 0; i < chars.length; i++){
	charArr.push(String.fromCharCode(chars[i]))
}
console.log("Chars: " + charArr)

var myLSTM = new nn.Architect.LSTM(chars.length,100,chars.length);

var learningRate = .3;


var one_hot = function(cc){
	var val = chars.indexOf(cc);
	onehot = new Array(chars.length).fill(0.0);
	onehot[val] = 1.0;
	return onehot;
}

var sample = function(LSTM, length, randomness) {
	var cc = 'a';
	var str = "";
	str += cc;
	for(var i = 0; i < length; i++) {
		
		var probs = LSTM.activate(one_hot(cc));

		var sum = 0;
		for(var ii = 0; ii < probs.length; ii++) {
			probs[i] = Math.pow(probs[i], randomness);
			sum += probs[i];
		}

		for(var ii = 0; ii < probs.length; ii++) {
			probs[i] /= sum;
		}

		var r = Math.random(); // returns [0,1]

		for (ii = 0 ; ii < probs.length && r >= probs[ii]; ii++);
		cc = chars[ii];
		console.log("Sampling: " + i + "/" + length + " :::: " + ii);
		console.log("%j", probs);
		str += String.fromCharCode(cc);
	}
	return str;
}




//ballpark estimates a cumulative probability
for (var i = 0; i < 2000; i++)
{
	console.log("Epoch " + i);

	for (var j = 0; j < txt.length; j++) {
		var cur = one_hot(txt[j]);
		console.log(j + " " + String.fromCharCode(txt[j]));

		if(j != 0) {
			myLSTM.propagate(0.3, cur);
		}
		distr = myLSTM.activate(cur);
		if ( (j + 1) % 1000 == 0) {
			console.log(sample(myLSTM, 50, 1) + "\n\n");
		}
	}
	
	console.log(sample(myLSTM, 50, 1) + "\n\n");
}


