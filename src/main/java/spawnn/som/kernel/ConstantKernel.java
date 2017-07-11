package spawnn.som.kernel;


public class ConstantKernel implements KernelFunction {
	
	private int radius;
	
	public ConstantKernel( int radius ) {
		this.radius = radius;
	}

	@Override
	public double getValue(double dist, double time) {
		if( dist <= radius )
			return 1;
		else
			return 0;
	}

}
