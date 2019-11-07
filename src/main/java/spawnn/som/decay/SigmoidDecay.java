package spawnn.som.decay;


public class SigmoidDecay extends DecayFunction {
	
	private double scale;
	
	public SigmoidDecay( double scale ) {
		this.scale = scale;
	}

	@Override
	public double getValue(double x) {
		
		return 1.0/(1.0+Math.exp( -x*scale+0.5*scale) );
	}
}




