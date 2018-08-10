package spawnn.som.grid;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

public class Grid2D_Map<T> extends Grid2D<T> {
	
	Map<GridPos,T> m = new HashMap<>();
	
	public Grid2D_Map( int xSize, int ySize ) { 
		super(xSize,ySize);
		for( int i = 0; i < xSize; i++ )
			for( int j = 0; j < ySize; j++ )
				setPrototypeAt( new GridPos(i,j), null);
	}
			
	// rook
	@Override
	public Collection<GridPos> getNeighbours( GridPos pos ) {
		Collection<GridPos> l = getNeighborCandidates(pos);
		l.retainAll(getPositions());
		return l;
	}	
	
	@Override
	public T getPrototypeAt(GridPos pos) {
		return m.get(pos);
	}

	@Override
	public T setPrototypeAt(GridPos pos, T v) {
		return m.put(pos, v);
	}
}
