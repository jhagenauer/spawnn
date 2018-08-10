package spawnn.som.kernel;

import spawnn.som.decay.DecayFunction;

public abstract class KernelFunction {
	
	protected DecayFunction df;	
	public KernelFunction( DecayFunction df ) {
		this.df = df;
	}
		
	public abstract double getValue( double dist, double x);
}
