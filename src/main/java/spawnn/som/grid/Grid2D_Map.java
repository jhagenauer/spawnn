package spawnn.som.grid;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class Grid2D_Map<T> extends Grid_Map<T> {
	
	public Grid2D_Map( int xSize, int ySize ) { 
		super();
		for( int i = 0; i < xSize; i++ )
			for( int j = 0; j < ySize; j++ )
				setPrototypeAt( new GridPos(i,j), null);
	}
		
	public Grid2D_Map(Map<GridPos,T> loadGrid) {
		super(loadGrid);
	}
			
	// in output space
	public int dist( GridPos aPos, GridPos bPos ) {
		return Math.abs( aPos.getPos(0)-bPos.getPos(0)) + Math.abs( aPos.getPos(1)-bPos.getPos(1) );
	}
	
	protected Collection<GridPos> getNeighborCandidates( GridPos pos ) {
		Set<GridPos> l = new HashSet<GridPos>();
		int x = pos.getPos(0);
		int y = pos.getPos(1);
		
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
}
