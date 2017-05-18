package spawnn.som.net;

import spawnn.UnsupervisedNet;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.decay.DecayFunction;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.KernelFunction;

public class SOM implements UnsupervisedNet {
			
	protected Grid<double[]> grid; 
	protected BmuGetter<double[]> bmuGetter;
	protected DecayFunction lr;
	protected KernelFunction nb;
		
	public SOM( KernelFunction nb, DecayFunction lr, Grid<double[]> grid, BmuGetter<double[]> bmuGetter ) {
		this.grid = grid;
		this.bmuGetter = bmuGetter;
		this.nb = nb;
		this.lr = lr;
	}
	
	@Override
	public void train( double t, double[] x) {
		GridPos bmuPos = bmuGetter.getBmuPos( x, grid );
		for( GridPos p : grid.getPositions() ) {
			double theta = nb.getValue( grid.dist( bmuPos, p ), t );
			double alpha = lr.getValue( t );
								
			double[] v = grid.getPrototypeAt(p);
			for( int j = 0; j < v.length; j++ )
				v[j] = v[j] + theta * alpha * (x[j] - v[j]);  
			grid.setPrototypeAt( p, v );
		}
	}
}
