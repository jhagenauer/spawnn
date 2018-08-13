package spawnn.som.grid;

import java.util.Collection;

public class Grid2DHex<T> extends Grid2D_Array<T> {
	
	public Grid2DHex( int xSize, int ySize ) {
		super( xSize, ySize );
	}
		
	@Override
	public int dist( GridPos aPos, GridPos bPos ) { 
		int x0 = aPos.getPos(0);
		int y0 = aPos.getPos(1);
		int x1 = bPos.getPos(0);
		int y1 = bPos.getPos(1);
		
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
		int x = pos.getPos(0);
		int y = pos.getPos(1);
			
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
