package nnet.activation;

public class Sigmoid implements Function {
	@Override
	public double f(double x) {
		return 1.0/(1.0+Math.exp(-x));
	}

}
