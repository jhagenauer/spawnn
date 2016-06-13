package nnet.activation;

public interface Function {
	public double f(double x);
	double fDevFOut(double x);
}
