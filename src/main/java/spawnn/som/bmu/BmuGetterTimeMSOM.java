package spawnn.som.bmu;

import java.util.Arrays;
import java.util.Set;

import spawnn.dist.Dist;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;

public class BmuGetterTimeMSOM extends BmuGetterContext {
	
	private Dist<double[]> fDist;
	private double alpha, beta;
	private double context[]; 
	private GridPos last;
		
	public BmuGetterTimeMSOM( Dist<double[]> fDist, double alpha, double beta ) {
		this.fDist = fDist;
		
		this.alpha = alpha;
		this.beta = beta;
		this.context = null;
	}
	
	@Override
	public GridPos getBmuPos(double[] x, Grid<double[]> grid, Set<GridPos> ign) {
		
		// get context
		if( last != null ) {
			double[] bmuV = grid.getPrototypeAt(last);
			context = new double[x.length];
			for( int i = 0; i < x.length; i++ )
				context[i] += (1.0 - beta) * bmuV[i];
			
			for( int i = 0; i < x.length; i++ )
				context[i] += beta * bmuV[x.length+i];
		}
		
		// search
		GridPos bmuPos = null;
		double minDist = Double.MAX_VALUE;
		for( GridPos p : grid.getPositions() ) {
			double[] v = grid.getPrototypeAt(p);
								
			double d;
			if( context != null ) {						
				double[] ci = Arrays.copyOfRange(v, x.length, v.length );								
				d = (1-alpha) * fDist.dist( x, v ) + alpha * fDist.dist( ci, context );
			} else
				d = fDist.dist( x, v );
			
			if( d < minDist ) {
				minDist = d;
				bmuPos = p;
			} 
		}
		
		// update context
		last = bmuPos;
		
		return bmuPos;
	}
	
	@Override
	public double[] getContext( double[] d ) { 
		return context;
	}
	
	public void setLastBmu(double[] d) { 
		this.context = d;
	}
	
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}
}
