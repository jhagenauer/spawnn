package spawnn.som.bmu;


import java.util.Set;

import spawnn.dist.Dist;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;

public class WeightedSumBmuGetter extends BmuGetter<double[]> {
	
	private double a,b;
	private Dist d1,d2;
	
	public WeightedSumBmuGetter( Dist d1, Dist d2, double a, double b ) {
		this.d1 = d1;
		this.d2 = d2;
		this.a = a;
		this.b = b;
	}
	
	@Override
	public GridPos getBmuPos( double[] x, Grid<double[]> grid, Set<GridPos> ign ) {
		
		double dist = Double.POSITIVE_INFINITY;
		GridPos bmu = null;
		
		for( GridPos p : grid.getPositions() ) {
			if( ign.contains(p) )
				continue;
			
			double[] v = grid.getPrototypeAt(p);
			double d = a * d1.dist(v,x) + b*d2.dist(v,x);
			if( d < dist ) {
				dist = d;
				bmu = p;
			}
		}
		return bmu;
	}

}
