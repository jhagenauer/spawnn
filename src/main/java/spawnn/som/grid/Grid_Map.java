package spawnn.som.grid;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public abstract class Grid_Map<T> extends Grid<T> {
	
	public abstract int dist( GridPos aPos, GridPos bPos );
	public abstract Collection<GridPos> getNeighbours( GridPos pos );

	protected Map<GridPos,T> grid = new HashMap<GridPos,T>();
	
	public Grid_Map() {}
			
	public Grid_Map( Map<GridPos,T> pos ) {
		for( GridPos p : pos.keySet() ) 
			setPrototypeAt(p, pos.get(p) );
	}
		
	// cache of sizes
	private Map<Integer,Integer> m = new HashMap<Integer,Integer>();
	
	public int getSizeOfDim( int n ) {
		if( !m.containsKey(n) ) {	
			int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;
			for( GridPos p : getPositions() ) {
				max = Math.max( max, p.getPos(n) );
				min = Math.min( min, p.getPos(n) );
			}
			m.put(n, max - min + 1);
		}		
		return m.get(n);
	}
	
	public Set<GridPos> getPositions() {
		return grid.keySet();
	}
		
	public int size() {
		return grid.size();
	}
		
	// slow
	public GridPos getPositionOf( T x ) {
		for( GridPos p : grid.keySet() )
			if( getPrototypeAt(p).equals(x) )
					return p;
		return null;
	}
	
	public T getPrototypeAt( GridPos pos ) {
		return grid.get(pos);
	}
		
	public T setPrototypeAt( GridPos pos, T v ) {
		m.clear(); 
		return grid.put( pos, v);
	}
		
	public Collection<T> getPrototypes() {
		return grid.values();
	}
	
	public Map<GridPos,T> getGridMap() {
		return grid;
	}
		
	@Override
	public String toString() {
		String s = "";
		for( GridPos p : getPositions()) 
			s+="["+p+","+Arrays.toString((double[])getPrototypeAt(p))+"]";	
		return s;
	}
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		
		for( GridPos p : grid.keySet() ) {
			result = result * prime + p.hashCode();/* + grid.get(p).hashCode()*/;
		}
				
		return result;
	}
	
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
						
		Grid_Map<T> other = (Grid_Map<T>) obj;
		
		if( other.size() != size() )
			return false;
				
		for( GridPos p : grid.keySet() ) 
			if( !other.getPositions().contains(p) || !getPrototypeAt(p).equals(other.getPrototypeAt(p)) )
				return false;
								
		for( GridPos p : other.getPositions() )
			if( !grid.containsKey(p) || !getPrototypeAt(p).equals(other.getPrototypeAt(p)) )
				return false;
		
		return true;
	}
}
