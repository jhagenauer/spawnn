package spawnn.som.kernel;

import spawnn.som.decay.DecayFunction;


public class BubbleKernel implements KernelFunction {
	
	private DecayFunction df;
		
	public BubbleKernel( DecayFunction df ) {
		this.df = df;
	}

	@Override
	public double getValue(double dist, double time) {
		if( dist <= df.getValue(time)*dist )
			return 1;
		else
			return 0;
	}

}
