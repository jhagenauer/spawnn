package spawnn.som.grid;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class Grid2D<T> extends Grid<T> {
	
	public Grid2D( int xSize, int ySize ) { 
		super();
		for( int i = 0; i < xSize; i++ )
			for( int j = 0; j < ySize; j++ )
				setPrototypeAt( new GridPos(i,j), null);
	}
		
	public Grid2D(Map<GridPos,T> loadGrid) {
		super(loadGrid);
	}
			
	// in output space
	public int dist( GridPos aPos, GridPos bPos ) {
		return Math.abs( aPos.getPosVector()[0]-bPos.getPosVector()[0]) + Math.abs( aPos.getPosVector()[1]-bPos.getPosVector()[1] );
	}
	
	protected Collection<GridPos> getNeighborCandidates( GridPos pos ) {
		Set<GridPos> l = new HashSet<GridPos>();
		int x = pos.getPosVector()[0];
		int y = pos.getPosVector()[1];
		
		l.add( new GridPos( x-1, y ) );
		l.add( new GridPos( x, y+1 ) );
		l.add( new GridPos( x+1, y ) );
		l.add( new GridPos( x, y-1 ) );
		return l;
	}
		
	// rook
	@Override
	public Collection<GridPos> getNeighbours( GridPos pos ) {
		Collection<GridPos> l = getNeighborCandidates(pos);
		l.retainAll(getPositions());
		return l;
	}
	
	@Override
	public String toString() {
		String s = "";
		for( int i = 0; i < getSizeOfDim(1); i++ ) {
			for( int j = 0; j < getSizeOfDim(0); j++ ) {
				GridPos p = new GridPos(j,i);
				T pt = getPrototypeAt(p);
				if( pt instanceof double[] )
					s+="("+p+")"+Arrays.toString((double[])getPrototypeAt(p))+", ";
				else
					s+="("+p+")"+p+", ";
			}
			s += "\n";
		}
		return s;
	}
	
}
