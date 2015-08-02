package spawnn.som.bmu;


import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import spawnn.ng.sorter.Sorter;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;

public class SorterBmuGetter<T> extends BmuGetter<T> {
	
	private Sorter<T> s;
	
	public SorterBmuGetter( Sorter<T> s ) {
		this.s = s;
	}

	@Override
	public GridPos getBmuPos( T x, Grid<T> grid, Set<GridPos> ign ) { // slow
		List<T> l = new ArrayList<T>(grid.getPrototypes());
		
		s.sort(x, l);
		
		for( int i = 0; i < l.size(); i++ ) {
			GridPos p = grid.getPositionOf(l.get(i));
			
			if( !ign.contains(p) )
				return p;
		}
		return null;
	}
}
