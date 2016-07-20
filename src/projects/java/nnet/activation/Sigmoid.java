package nnet.activation;

public class Sigmoid implements Function { // aka logistic function
	@Override
	public double f(double x) {
		return 1.0/(1.0+Math.exp(-x));
	}
	
	@Override
	public double fDevFOut(double fOut ) {
		return fOut * (1.0 - fOut);
	}

}
