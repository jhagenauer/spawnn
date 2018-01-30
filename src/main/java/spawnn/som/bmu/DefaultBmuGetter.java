package spawnn.som.bmu;


import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import spawnn.dist.Dist;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;

public class DefaultBmuGetter<T> extends BmuGetter<T> {
	
	private Dist<T> di;
	
	public DefaultBmuGetter( Dist<T> d ) {
		this.di = d;
	}

	@Override
	public GridPos getBmuPos( T x, Grid<T> grid, Set<GridPos> ign ) {
		double dist = Double.POSITIVE_INFINITY;
		
		GridPos bmu = null; 
		List<GridPos> gp = new ArrayList<>(grid.getPositions());
		Collections.shuffle(gp);
		for( GridPos p : grid.getPositions() ) {
			
			if( ign != null && ign.contains(p) )
				continue;
			
			T v = grid.getPrototypeAt(p);			
			double d = di.dist( v, x );		
			if( d < dist ) {
				dist = d;
				bmu = p;	
			}
		}
		
		if( bmu == null ) 
			throw new RuntimeException("No bmu found");
							
		return bmu;
	}
}
