package spawnn.som.kernel;

import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.LinearDecay;

public class GaussKernel extends KernelFunction {
		
	private double min = Math.pow(10,-127);
	
	public GaussKernel( int max ) {
		super( new LinearDecay(max, 0) );
	}
	
	public GaussKernel( DecayFunction df ) {
		super(df);
	}
		
	@Override
	public double getValue(double dist, double time) {
		double sigma = Math.max( df.getValue(time), min );					
		return Math.max( Math.exp(  -0.5 * Math.pow( dist / sigma , 2 ) ), 0 );
	}
}
