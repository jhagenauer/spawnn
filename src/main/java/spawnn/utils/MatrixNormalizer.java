package spawnn.utils;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.jblas.DoubleMatrix;

public class MatrixNormalizer extends Normalizer {
	
	Transform[] tt;
	SummaryStatistics[][] ds;
	
	public MatrixNormalizer( Transform[] tt, DoubleMatrix X, boolean lastIc ) {
		this.tt = tt;
		
		this.ds = new SummaryStatistics[tt.length][X.columns - (lastIc ? 1 : 0)];
		for( int i = 0; i < tt.length; i++ ) {
			// calculate summary statistics before applying t
			for( int j = 0; j < ds[i].length; j++ ) {
				ds[i][j] = new SummaryStatistics();
				for( int k = 0; k < X.rows; k++  )
					ds[i][j].addValue(X.get(k,j));			
			}
			
			// normalize
			normalize(X, i);			
		}
	}
	
	public void normalize(DoubleMatrix X) {	
		for( int i = 0; i < tt.length; i++ ) 
			normalize(X, i );								
	}
	
	private void normalize(DoubleMatrix X, int i ) {
		Transform t = tt[i];
		for( int j = 0; j < ds[i].length; j++ )
			for (int k = 0; k < X.rows; k++ ) {
				double d = X.get(k,j);
				if (t == Transform.zScore) 
					d = (d - ds[i][j].getMean()) / ds[i][j].getStandardDeviation();
				else if (t == Transform.scale01)
					d = (d - ds[i][j].getMin()) / (ds[i][j].getMax() - ds[i][j].getMin());
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
				X.put(k,j,d);
			}		
	}
}
