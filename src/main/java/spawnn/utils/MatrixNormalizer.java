package spawnn.utils;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.jblas.DoubleMatrix;

public class MatrixNormalizer extends Normalizer {
	
	Transform[] tt;
	SummaryStatistics[] ds;
	
	public MatrixNormalizer( Transform[] t, DoubleMatrix X ) {
		
		ds = new SummaryStatistics[X.columns];
		for (int i = 0; i < ds.length; i++)
			ds[i] = new SummaryStatistics();
		
		for( int i = 0; i < X.rows; i++ )
			for( int j = 0; j < X.columns; j++ ) {
				double v = X.get(i, j);
				if( !Double.isNaN(v) )
					ds[j].addValue(v);
			}
		
		this.tt = t;
		normalize(X);
	}
	
	public void normalize(DoubleMatrix X) {
		for (int i = 0; i < X.rows; i++) 
			for( int j = 0; j < X.columns; j++ ) {
			double d = X.get(i,j);
			for( Transform t : tt  ) {
				if (t == Transform.zScore) {
					if( ds[j].getStandardDeviation() != 0 )
						d = (d - ds[j].getMean()) / ds[j].getStandardDeviation();
				} else if (t == Transform.scale01)
					d = (d - ds[j].getMin()) / (ds[j].getMax() - ds[j].getMin());
				else if( t == Transform.sqrt )
					d = Math.sqrt(d);
				else if( t == Transform.log )
					d = Math.log(d);
				else if( t == Transform.log1 )
					d = Math.log( d+1.0 );
				else if( t == Transform.none )
					;
				else
					throw new RuntimeException(t+" not supported!");
			}
			X.put(i, j, d);
		}
	}
}
