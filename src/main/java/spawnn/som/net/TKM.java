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

public class TKM {
			
	protected Grid<double[]> grid; 
	protected DecayFunction lr;
	protected KernelFunction nb;
	private List<Map<GridPos,Double>> pots;
	
	
	public TKM( KernelFunction nb, DecayFunction lr, Grid<double[]> grid ) {
		this.grid = grid;
		this.nb = nb;
		this.lr = lr;
		this.pots = new ArrayList<Map<GridPos,Double>>();
	}
	
	private double getPotential( int t, double d, GridPos p ) {
		if( t >= 0  )
			return d * getPotential( t-1, d, p ) - 0.5 * Math.pow( pots.get(t).get(p), 2);
		else
			return 0;
	}
				
	public void train( double t, double[] x) {
		
		// update activations
		Dist eDist = new EuclideanDist();
		double d = 0.5;
		
		// update cur potential potential
		Map<GridPos,Double> curPot = new HashMap<GridPos,Double>();
		for( GridPos p : grid.getPositions() ) {
			double[] v = grid.getPrototypeAt(p);
			curPot.put(p, eDist.dist(v,x) );
		}
		pots.add( curPot );
		
		// get BMU	
		GridPos bmuPos = null;
		double max = Double.MIN_VALUE;
		for( GridPos p : grid.getPositions() ) {
			double pot = getPotential(pots.size()-1, d, p);
			if( pot > max ) {
				bmuPos = p;
				max = pot;
			}
		}
				
		// update
		for( GridPos p : grid.getPositions() ) {
			double theta = nb.getValue( grid.dist( bmuPos, p ), t );
			double alpha = lr.getValue( t );
								
			double[] v = grid.getPrototypeAt(p);
			for( int j = 0; j < v.length; j++ )
				v[j] = v[j] + theta * alpha * (x[j] - v[j]);  
			grid.setPrototypeAt( p, v );
		}
	}
		
	public Grid<double[]> getGrid() {
		return grid;
	}
}
