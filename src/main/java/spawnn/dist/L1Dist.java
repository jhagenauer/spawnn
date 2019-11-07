package spawnn.dist;

public class L1Dist implements Dist<double[]> {
	
	private int[] idx;

	public L1Dist() {
		this.idx = null;
	}

	public L1Dist(int[] idx) {
		this.idx = idx;
	}
	
	@Override
	public double dist(double[] a, double[] b) {
		double d = 0;
		if( idx == null )
			for( int i = 0; i < a.length; i++ )
				d += Math.abs( a[i]-b[i] );
		else {
			for( int i : idx )
				d += Math.abs( a[i]-b[i] );
		}
		return d;	
	}
}
