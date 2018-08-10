package spawnn.som.kernel;

import spawnn.som.decay.DecayFunction;

public class LinearKernel extends KernelFunction {
	
	public LinearKernel( DecayFunction df ) {
		super(df);
	}

	@Override
	public double getValue(double dist, double radius) {
		return Math.max( 1.0 - dist*df.getValue(radius), 0);
	}

}
