package spawnn.som.bmu;


import java.util.HashSet;
import java.util.Set;

import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;

public abstract class BmuGetter<T> {
	public GridPos getBmuPos( T x, Grid<T> grid ) {
		return getBmuPos(x, grid, new HashSet<GridPos>() );
	}
	
	public abstract GridPos getBmuPos( T x, Grid<T> grid, Set<GridPos> ign );
}
