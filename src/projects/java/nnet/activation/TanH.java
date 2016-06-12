package nnet.activation;

public class TanH implements Function {
	@Override
	public double f(double x) {
		return Math.tanh(x);
	}

}
