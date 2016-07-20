package nnet.activation;

public class SoftMax implements Function {

	public double f(double[] x, int i) {
		double sum = 0;
		for( double d : x )
			sum += Math.exp(d);
		return Math.exp(x[i])/sum;
	}
	
	@Override
	public double fDevFOut(double fOut ) {
		return fOut * (1.0 - fOut);
		//throw new RuntimeException("Don't call me!");
	}
	
	// cross-entropy cost model, only last layer
	public double fDevFOut(double[] x, int i, double[] desired ) {
		return x[i] - desired[i];
	}

	@Override
	public double f(double x) {
		throw new RuntimeException("Don't call me!");
	}
}
