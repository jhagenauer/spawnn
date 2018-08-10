package spawnn.som.grid;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public abstract class Grid<T> {
	
	public abstract int dist( GridPos aPos, GridPos bPos );
	public abstract Collection<GridPos> getNeighbours( GridPos pos );
	public abstract Set<GridPos> getPositions();
	public abstract int size();
	public abstract GridPos getPositionOf( T x );
	public abstract T getPrototypeAt( GridPos pos );
	public abstract T setPrototypeAt( GridPos pos, T v );
	public abstract Collection<T> getPrototypes();
	
	@Deprecated
	public abstract Map<GridPos,T> getGridMap();
	
	public int getMaxDist() {
		int max = 0;
		for( GridPos p1 : getPositions() )
			for( GridPos p2 : getPositions() )
				if( dist(p1,p2) > max )
					max = dist(p1,p2);
		return max;
	}
	
	public int getNumDimensions() {
		return getPositions().iterator().next().length();
	}
	
	public int getSizeOfDim( int n ) {
		int min = 0, max = Integer.MAX_VALUE;
		for( GridPos p : getPositions() ) {
			max = Math.max( max, p.getPos(n) );
			min = Math.min( min, p.getPos(n) );
		}
		return  max - min + 1;
	}	
	
	// contiguity map of prototypes
	public Map<T,List<T>> getContiguityMap() {
		Map<T,List<T>> m = new HashMap<T,List<T>>();
		for( GridPos gp : getPositions() ) {
			List<T> l = new ArrayList<T>();
			for( GridPos nb : getNeighbours(gp))
				l.add( getPrototypeAt(nb));
			m.put( getPrototypeAt(gp), l);
		}
		return m;
	}
}
