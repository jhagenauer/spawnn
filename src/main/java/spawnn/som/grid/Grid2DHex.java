package spawnn.som.grid;

import java.util.Collection;
import java.util.Map;

public class Grid2DHex<T> extends Grid2D<T> {
	
	public Grid2DHex( int xSize, int ySize ) {
		super( xSize, ySize );
	}
	
	public Grid2DHex(Map<GridPos,T> loadGrid) {
		super(loadGrid);
	}
	
	@Override
	public int dist( GridPos aPos, GridPos bPos ) { 
		int x0 = aPos.getPosVector()[0];
		int y0 = aPos.getPosVector()[1];
		int x1 = bPos.getPosVector()[0];
		int y1 = bPos.getPosVector()[1];
		
		int xDist = Math.abs( x0-x1 );
		int yDist = Math.abs( y0-y1 );
		
		return getHexDist(xDist, yDist, y1 > y0, x0 % 2 == 0 );
	}
	

	protected int getHexDist( int minXDist, int minYDist, boolean up, boolean evenX ) {
		int dist = minXDist + minYDist;
		if( up ) { // up
			if( evenX )
				dist -= (int)Math.min( minYDist, minXDist/2 );
			else 
				dist -= (int)Math.min( minYDist, Math.ceil( (double)minXDist/2 ) );
		} else { // down
			if( evenX )
				dist -= (int)Math.min( minYDist, Math.ceil( (double)minXDist/2 ) );
			else 
				dist -= (int)Math.min( minYDist, minXDist/2 );
		}
		return dist;
	}
	
	@Override
	protected Collection<GridPos> getNeighborCandidates( GridPos pos ) {
		Collection<GridPos> l = super.getNeighborCandidates(pos);
		int x = pos.getPosVector()[0];
		int y = pos.getPosVector()[1];
			
		if( x % 2 == 0 ) {
				l.add( new GridPos( x-1, y-1 ) );
				l.add( new GridPos( x+1, y-1 ) );
		} else if( x % 2 == 1  ) {
				l.add( new GridPos( x-1, y+1 ) );
				l.add( new GridPos( x+1, y+1 ) );
		} 
		
		return l;
	}
		
	@Override
	public Collection<GridPos> getNeighbours( GridPos pos ) { 
		Collection<GridPos> l = getNeighborCandidates(pos);
		l.retainAll(getPositions()); // only positions that are present in the grid
						
		return l;
	}
}
