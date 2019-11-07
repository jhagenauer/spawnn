package spawnn.som.net;


import java.util.HashMap;
import java.util.Map;

import spawnn.UnsupervisedNet;
import spawnn.dist.Dist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.decay.DecayFunction;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;
import spawnn.som.grid.GrowingGrid2D;
import spawnn.som.kernel.KernelFunction;

// See Fritzke 95, requires finetuning with regular SOM
public class GrowingSOM implements UnsupervisedNet {
			
	protected GrowingGrid2D grid; 
	protected BmuGetter<double[]> bmuGetter;
	protected KernelFunction nb;
	protected DecayFunction lr;
	private int insPeriod, c;
	private Map<GridPos,Integer> counter;
	private Dist d;
	private double maxShare;
	
	public GrowingSOM( KernelFunction nb, DecayFunction lr, GrowingGrid2D grid, BmuGetter<double[]> bmuGetter, Dist d, int insPeriod, double maxShare ) {
		this.grid = grid;
		this.bmuGetter = bmuGetter;
		this.nb = nb;
		this.lr = lr;
		this.counter = new HashMap<GridPos,Integer>();
		this.insPeriod = insPeriod;
		this.maxShare = maxShare;
		this.d = d;
		this.c = 0;
	}
				
	public void train( double t, double[] x ) {
		
		GridPos bmuPos = bmuGetter.getBmuPos( x, grid );
		if( counter.containsKey( bmuPos) )
			counter.put( bmuPos, counter.get(bmuPos)+1 );
		else
			counter.put( bmuPos, 1);
		
		if( c == grid.getPositions().size() * insPeriod ) { // k x m x l
			
			// get unit with most hits, other criteria are possible
			GridPos insPos = null;
			int max = Integer.MIN_VALUE;
			for( GridPos pos : counter.keySet() ) {
				int v = counter.get(pos);
				if( v > max ) {
					max = v;
					insPos = pos;
				}
			}
			double[] vec = grid.getPrototypeAt(insPos);
			
			
			int sum = 0;
			for( int v : counter.values() ) 
				sum += v;			
			
			// abort training
			/*if( (double)max/sum <= maxShare )
				return false;*/
			
			if( grid.size() >= 500 )
				return;
													
			GridPos worstNb = null;
			double dist = Double.MIN_VALUE;
			for( GridPos nb : grid.getNeighbours(insPos) ) {
				double[] nbv = grid.getPrototypeAt(nb);
				if( d.dist( vec, nbv ) > dist ) {
					dist = d.dist(vec, nbv);
					worstNb = nb;
				}
			}
			
			// get pos and dim
			int dim = -1;
			int pos = -1;
			for( int k = 0; k < insPos.length(); k++ ) {
				if( worstNb.getPos(k) != insPos.getPos(k) ) {
					dim = k;
					if( worstNb.getPos(k) < insPos.getPos(k) )
						pos = worstNb.getPos(k);
					else
						pos = insPos.getPos(k);
					break;
				}
			}
			grid.addVector(dim, pos);		
			
			
			c = 0;
			counter = new HashMap<GridPos, Integer>();	
		} 
													
		for( GridPos p : grid.getPositions() ) {
			int gDist = grid.dist( bmuPos, p );
														
			double[] v = grid.getPrototypeAt(p);
			for( int j = 0; j < v.length; j++ )
				v[j] = v[j] + lr.getValue(t) * nb.getValue( gDist, t) * (x[j] - v[j]);
			
			grid.setPrototypeAt( p, v );
		}
		c++;
		return;
	}
		
	public Grid<double[]> getGrid() {
		return grid;
	}
	
	public BmuGetter<double[]> getBmuGetter() {
		return bmuGetter;
	}
	
	public double[] getBmuVector( double[] x ) {
		return grid.getPrototypeAt( bmuGetter.getBmuPos(x, grid) );
	}
}
