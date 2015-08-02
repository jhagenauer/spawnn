package spawnn.som.bmu;

import java.util.HashSet;
import java.util.Set;

import spawnn.dist.Dist;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;

public class KangasBmuGetter<T> extends BmuGetter<T> {
	
	private int k;
	private BmuGetter<T> a,b;
		
	public KangasBmuGetter( Dist<T> da, Dist<T> db, int k ) {
		this( new DefaultBmuGetter<T>(da), new DefaultBmuGetter<T>(db), k);
	}
	
	public KangasBmuGetter( BmuGetter<T> a, BmuGetter<T> b, int k ) {
		this.a = a;
		this.b = b;
		this.k = k;
	}
	
	@Override
	public GridPos getBmuPos(T x, Grid<T> grid, Set<GridPos> ign ) {
		GridPos bmuA = a.getBmuPos(x, grid);
		if( k == 0 ) 
			return bmuA;
		
		Set<GridPos> ignore = new HashSet<GridPos>(ign);
		for( GridPos p : grid.getPositions() ) 
			if( grid.dist(bmuA, p) > k )
				ignore.add(p);
					
		return b.getBmuPos(x, grid, ignore);
	}

	@Deprecated
	public GridPos getBmuBPos(T x, Grid<T> grid, Set<GridPos> ign, GridPos bmuA ) {
		Set<GridPos> ignore = new HashSet<GridPos>();
		if( ign != null )
			ignore.addAll(ign);
		for( GridPos p : grid.getPositions() ) 
			if( grid.dist(bmuA, p) > k )
				ignore.add(p);
					
		return b.getBmuPos(x, grid, ignore);
	}
	
	@Deprecated
	public GridPos getBmuAPos( T x, Grid<T> grid, Set<GridPos> ign ) {
		return a.getBmuPos(x, grid, ign);
	}
	
	public int getRadius() {
		return k;
	}
}
