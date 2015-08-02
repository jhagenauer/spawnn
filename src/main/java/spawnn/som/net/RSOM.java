package spawnn.som.net;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.decay.DecayFunction;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.KernelFunction;

public class RSOM {
			
	protected Grid<double[]> grid; 
	protected DecayFunction lr;
	protected KernelFunction nb;
	private List<Map<GridPos,double[]>> pots;
	
	
	public RSOM( KernelFunction nb, DecayFunction lr, Grid<double[]> grid ) {
		this.grid = grid;
		this.nb = nb;
		this.lr = lr;
		this.pots = new ArrayList<Map<GridPos,double[]>>();
	}
	
	private double[] getY( int t, double alpha, GridPos p ) {
		int length = pots.get(0).keySet().iterator().next().getPosVector().length;
		if( t >= 0  ) {
			double[] a = getY( t-1, alpha, p );
			double[] b = pots.get(t).get(p);
			
			double[] r = new double[length];
			
			for( int i = 0; i < r.length; i++ )
				r[i] = (1-alpha) * a[i] + alpha * b[i];
			return r;
		} else
			return new double[length];
	}
	
	
				
	public void train( double t, double[] x) {
		
		// update activations
		Dist eDist = new EuclideanDist();
		double alpha = 0.5;
		
		// update cur potential potential
		Map<GridPos,double[]> curY = new HashMap<GridPos,double[]>();
		for( GridPos p : grid.getPositions() ) {
			
			double[] v = grid.getPrototypeAt(p);
			double[] d = new double[v.length];
			
			for( int j = 0; j < d.length; j++ )
				d[j] = x[j] - v[j];  
			curY.put(p, d );
		}
		pots.add( curY );
		
		// get BMU	
		GridPos bmuPos = null;
		double min = Double.MAX_VALUE;
		for( GridPos p : grid.getPositions() ) {
			double[] y = getY(pots.size()-1, alpha, p);
			double[] v = new double[y.length];
			
			double pot = Math.pow( eDist.dist(y, v), 2);
			if( pot < min ) {
				bmuPos = p;
				min = pot;
			}
		}
				
		// update
		for( GridPos p : grid.getPositions() ) {
			double theta = nb.getValue( grid.dist( bmuPos, p ), t );
			double lRate = lr.getValue( t );
								
			double[] v = grid.getPrototypeAt(p);
			for( int j = 0; j < v.length; j++ )
				v[j] = v[j] + theta * lRate * min;  
			grid.setPrototypeAt( p, v );
		}
	}
		
	public Grid<double[]> getGrid() {
		return grid;
	}
}
