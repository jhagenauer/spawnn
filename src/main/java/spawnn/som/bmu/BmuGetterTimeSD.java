package spawnn.som.bmu;

import java.util.Set;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;

public class BmuGetterTimeSD extends BmuGetterContext {
	
	private Dist<double[]> fDist;
	private EuclideanDist eDist;
	private double alpha;
	private GridPos last; 
	private double[] context;
		
	public BmuGetterTimeSD( Dist<double[]> fDist, double alpha ) {
		this.fDist = fDist;
		this.eDist = new EuclideanDist();
		
		this.alpha = alpha;
		this.last = null;
	}
	
	@Override
	public GridPos getBmuPos(double[] x, Grid<double[]> grid, Set<GridPos> ign) {
		int numDim = grid.getNumDimensions();
		int fullLength = grid.getPrototypes().iterator().next().length;
		
		if( last != null ) {
			int[] bmuPV = last.getPosVector();
			context = new double[bmuPV.length];
			for( int i = 0; i < bmuPV.length; i++ )
				context[i] = bmuPV[i];
		}
		
		GridPos bmuPos = null;
		double minDist = Double.MAX_VALUE;
		for( GridPos p : grid.getPositions() ) {
			double[] v = grid.getPrototypeAt(p);
								
			double d;
			if( context != null )
				d = (1-alpha) * fDist.dist( x, v ) + alpha * eDist.dist( v, fullLength - numDim, context, 0 );
			else
				d = (1-alpha) * fDist.dist( x, v );
			
			if( d < minDist ) {
				minDist = d;
				bmuPos = p;
			} 
		}
				
		last = bmuPos;
		
		return bmuPos;
	}
	
	@Override
	public double[] getContext(double[] d) { // d is only dummy-variable, context does not depend on d
		return context;
	}
	
	public void setLastBmu(GridPos d) { 
		this.last = d;
	}
	
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}
}
