package nnet.activation;

public class Identity implements Function {
	@Override
	public double f(double x) {
		return x;
	}

	@Override
	public double fDevFOut(double fOut) {
		return 1.0;
	}

}
