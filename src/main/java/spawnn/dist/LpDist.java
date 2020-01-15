package spawnn.dist;

public class LpDist implements Dist<double[]> {
	
	private int[] idx;
	private int p;

	public LpDist(int p) {
		this.idx = null;
		this.p = p;
	}

	public LpDist(int[] idx, int p ) {
		this.idx = idx;
		this.p = p;
	}
	
	@Override
	public double dist(double[] a, double[] b) {
		double d = 0;
		if( idx == null )
			for( int i = 0; i < a.length; i++ )
				d += Math.pow( Math.abs( a[i]-b[i] ), p );
		else {
			for( int i : idx )
				d += Math.pow( Math.abs( a[i]-b[i] ), p );
		}
		
		return Math.pow(d, 1.0/p);
	}
}
