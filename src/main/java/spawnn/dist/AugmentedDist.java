package spawnn.dist;


public class AugmentedDist implements Dist<double[]> {
	
	private int[] ga, fa;
	private double alpha;

	public AugmentedDist(int[] ga, int[] fa, double alpha) {
		this.ga = ga;
		this.fa = fa;
		this.alpha = alpha;
	}
	
	@Override
	public double dist(double[] a, double[] b) {		
		double dist = 0;
		for( int i : ga )
			dist += ( alpha * ( a[i] - b[i] ) ) * ( alpha * ( a[i] - b[i] ) );
		for( int i : fa )
			dist += (a[i] - b[i])*(a[i] - b[i]);					
		return Math.sqrt( dist );
	}
}
