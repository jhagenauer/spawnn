package spawnn.dist;


public class SquaredDist implements Dist<double[]> {
	
	private Dist<double[]> d;
	private double pow;

	public SquaredDist(Dist<double[]> d, double pow ) {
		this.d = d;
		this.pow = pow;
	}
	
	@Override
	public double dist(double[] a, double[] b) {		
		return Math.pow( d.dist(a, b), pow);
	}
}
