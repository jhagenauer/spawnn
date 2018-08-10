package spawnn.som.grid;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

public abstract class Grid2D<T> extends Grid<T> {
	
	private Set<GridPos> positions = new HashSet<>();
	private int xSize, ySize;
		
	public Grid2D( int xSize, int ySize ) { 
		super();
		for( int i = 0; i < xSize; i++ )
			for( int j = 0; j < ySize; j++ )
				positions.add( new GridPos(i,j) );
		this.xSize = xSize;
		this.ySize = ySize;
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

	@Override
	public Set<GridPos> getPositions() {
		return positions;
	}

	@Override
	public int size() {
		return xSize*ySize;
	}
}
