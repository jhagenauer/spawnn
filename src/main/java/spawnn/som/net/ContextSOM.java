package spawnn.som.net;

import spawnn.som.bmu.BmuGetterContext;
import spawnn.som.decay.DecayFunction;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.KernelFunction;

public class ContextSOM extends SOM {
			
	public ContextSOM(KernelFunction nb, DecayFunction lr, Grid<double[]> grid, BmuGetterContext bmuGetter, int sampleLength ) {
		super(nb, lr, grid, bmuGetter);
	}

	@Override
	public void train( double t, double[] x) {
		double[] context = ((BmuGetterContext)bmuGetter).getContext(x);
		GridPos bmuPos = bmuGetter.getBmuPos( x, grid );	
																						
		for( GridPos p : grid.getPositions() ) {
			double theta = nb.getValue( grid.dist( bmuPos, p ), t );
			double alpha = lr.getValue( t );
			double adapt = theta * alpha;
				
			// adapt weights
			double[] w = grid.getPrototypeAt(p);				
			for( int i = 0; i < x.length; i++ )  
				w[i] += adapt * (x[i] - w[i]);
			
			// adapt context
			if( context != null )
				for( int i = 0; i < context.length; i++ ) 
					w[ x.length+ i ] += adapt * (context[i] - w[ x.length + i]);
										
			grid.setPrototypeAt( p, w );
		}
	}
}
