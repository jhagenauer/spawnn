package spawnn.dist;


public class VecDist implements Dist<double[]> {
	
	private int t;
	
	public VecDist( int t ) {
		super();
		this.t = t;
	}

	@Override
	public double dist(double[] a, double[] b) {
				
		// euclidean
		Dist<double[]> di = new EuclideanDist();
		double ed1 = di.dist( new double[]{ a[0],a[1] } , new double[]{ b[0], b[1]} );
		double ed2 = di.dist( new double[]{ a[2],a[3] } , new double[]{ b[2], b[3]} );
		
		// manhattan
		double hd1 = Math.abs( a[0] - b[0] ) + Math.abs(a[1] - b[1] );
		double hd2 = Math.abs( a[2] - b[2] ) + Math.abs(a[3] - b[3] );
		
		switch(t) {
		case 0:
			return di.dist(a, b);
		case 1:
			return ed1 + ed2;
		case 2:
			return Math.sqrt( ed1*ed1 + ed2*ed2 );
		case 3:
			return Math.max( ed1, ed2);
		case 4:
			return hd1+hd2;
		case 5:
			return Math.sqrt( hd1*hd1 + hd2*hd2 );
		case 6:
			return Math.max( hd1, hd2);	
					
		default:
			return -1;
		}
	}
}
