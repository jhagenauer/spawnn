package spawnn.som.grid;

import java.util.Collection;

public class Grid2DToroid<T> extends Grid2D_Map<T> {
	
	public Grid2DToroid( int xSize, int ySize ) {
		super(xSize, ySize);
	}
	
	// in output space
	public int dist( GridPos aPos, GridPos bPos ) {
		int x0 = aPos.getPos(0);
		int y0 = aPos.getPos(1);
		int x1 = bPos.getPos(0);
		int y1 = bPos.getPos(1);
		
		int xDirectDist = Math.abs( x0-x1 );
		int yDirectDist = Math.abs( y0-y1 );
		int xNonDirectDist = Math.min( Math.abs( x0 + getSizeOfDim(0)- x1 ), Math.abs( x1 + getSizeOfDim(0)- x0 ));
		int yNonDirectDist = Math.min( Math.abs( y0 + getSizeOfDim(1) - y1 ), Math.abs( y1 + getSizeOfDim(1) - y0 ) );
		
		return Math.min(xDirectDist, xNonDirectDist) + Math.min(yDirectDist, yNonDirectDist);
	}
	
	public Collection<GridPos> getNeighbours( GridPos pos ) {
		Collection<GridPos> l = super.getNeighbours(pos);
		
		int x = pos.getPos(0);
		int y = pos.getPos(1);
		
		int xSize = getSizeOfDim(0);
		int ySize = getSizeOfDim(1);
		if( x == 0 )
			l.add( new GridPos( xSize - 1, y ) );
		if( x == xSize -1 )
			l.add( new GridPos( 0, y ) );
		
		if( y == 0 )
			l.add( new GridPos( x, ySize - 1 ) );
		if( y == ySize -1 )
			l.add( new GridPos( x, 0 ) );	
		return l;
	}
}

