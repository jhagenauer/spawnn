package nnet.activation;

public class TanH implements Function {
	@Override
	public double f(double x) {
		return Math.tanh(x);
	}

	@Override
	public double fDevFOut(double fOut) {
		return 1.0 - Math.pow(fOut, 2);
	}

}
