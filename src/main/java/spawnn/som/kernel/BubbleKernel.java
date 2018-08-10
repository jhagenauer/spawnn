package spawnn.som.kernel;

import spawnn.som.decay.DecayFunction;


public class BubbleKernel extends KernelFunction {
		
	public BubbleKernel( DecayFunction df ) {
		super(df);
	}

	@Override
	public double getValue(double dist, double radius) {
		if( dist <= df.getValue(radius) ) 
			return 1;
		else
			return 0;
	}

}
