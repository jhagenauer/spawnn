package spawnn.dist;

public class TestDist implements Dist<double[]> {
		
	@Override
	public double dist(double[] a, double[] b) {
		double d = 0;
		for( int i = 0; i < a.length; i++ )
			if( i == 0 )
				d += 0.0*Math.pow( a[i]-b[i],2 );
			else
				d += Math.pow( a[i]-b[i],2 );
		return Math.sqrt(d);	
	}
}
