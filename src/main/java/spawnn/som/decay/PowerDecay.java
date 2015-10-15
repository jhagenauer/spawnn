package spawnn.som.decay;


// From Ritter, Martinetz and Schulten... primarily used for neural gas 
public class PowerDecay extends DecayFunction {
	
	private double i, f;
	
	public PowerDecay( double i, double f ) {
		this.i = i;
		this.f = f;
	}

	@Override
	public double getValue(double x) {
		return i*Math.pow(f/i,x);
	}
}




