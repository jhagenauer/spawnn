package spawnn.som.decay;

public class ConstantDecay extends DecayFunction {
	
	private double c;
	
	public ConstantDecay(double c ) {
		this.c = c;
	}

	@Override
	public double getValue(double x) {
		return c;
	}

}
