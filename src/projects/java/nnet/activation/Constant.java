package nnet.activation;

public class Constant implements Function {
	private double x;
	public Constant( double x ) {
		this.x = x;
	}
	
	@Override
	public double f(double x) {
		return this.x;
	}

	@Override
	public double fDevFOut(double fOut) {
		return 0;
	}
}
