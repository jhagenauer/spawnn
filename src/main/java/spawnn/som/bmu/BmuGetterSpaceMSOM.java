package spawnn.som.bmu;

import java.util.Map;
import java.util.Set;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;

public class BmuGetterSpaceMSOM extends BmuGetterContext {
	
	private Dist<double[]> fDist;
	private double alpha, beta;
	private Map<double[],double[]> bmuHist; 
	Map<double[],Map<double[],Double>> weightMatrix;
		
	public boolean bmuHistMutable = false; 	
	
	public BmuGetterSpaceMSOM( Map<double[],double[]> bmuHist, Map<double[],Map<double[],Double>> weightMatrix, Dist<double[]> fDist, double alpha, double beta ) {					
			this.fDist = fDist;
			this.weightMatrix = weightMatrix;
					
			this.alpha = alpha;
			this.beta = beta;
			this.bmuHist = bmuHist;
	}
	
	@Override
	public GridPos getBmuPos(double[] x, Grid<double[]> grid, Set<GridPos> ign) {
		
		// get context
		double[] context = getContext(x);
										
		// search
		GridPos bmuPos = null;	
		double minDist = Double.MAX_VALUE;
		for( GridPos p : grid.getPositions() ) {
			double[] v = grid.getPrototypeAt(p);
											
			double d = (1-alpha) * fDist.dist( x, v ) + alpha * ((EuclideanDist)fDist).dist( v, x.length, context, 0 );
			
			if( d < minDist ) {
				minDist = d;
				bmuPos = p;
			}
		}
		
		// update hist
		if( bmuHistMutable ) 
			bmuHist.put(x, grid.getPrototypeAt(bmuPos) );	
		
		return bmuPos;
	}
	
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}
	
	@Override
	public double[] getContext(double[] x ) {
		Map<double[],Double> nbs = weightMatrix.get(x);
				
		double sumWeights = 0;
		for( double w : nbs.values() )
			sumWeights += w;

		double[] context = new double[x.length];
		for( double[] v : nbs.keySet() ) {
			
			double wi = nbs.get(v); // weight v/x
			double[] p = bmuHist.get(v);
			
			for( int i = 0; i < x.length; i++ )
				context[i] += (1.0 - beta) * wi * p[i] / sumWeights;
			
			for( int i = 0; i < x.length; i++ )
				context[i] += beta * wi * p[x.length+i] / sumWeights;
		}
				
		return context;	
	}
}
