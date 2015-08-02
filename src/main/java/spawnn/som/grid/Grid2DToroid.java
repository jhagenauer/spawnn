package spawnn.som.grid;

import java.util.Collection;
import java.util.Map;

public class Grid2DToroid<T> extends Grid2D<T> {
	
	public Grid2DToroid( int xSize, int ySize ) {
		super(xSize, ySize);
	}
					
	public Grid2DToroid(Map<GridPos, T> loadGrid) {
		super(loadGrid);
	}

	// in output space
	public int dist( GridPos aPos, GridPos bPos ) {
		int x0 = aPos.getPosVector()[0];
		int y0 = aPos.getPosVector()[1];
		int x1 = bPos.getPosVector()[0];
		int y1 = bPos.getPosVector()[1];
		
		int xDirectDist = Math.abs( x0-x1 );
		int yDirectDist = Math.abs( y0-y1 );
		int xNonDirectDist = Math.min( Math.abs( x0 + getSizeOfDim(0)- x1 ), Math.abs( x1 + getSizeOfDim(0)- x0 ));
		int yNonDirectDist = Math.min( Math.abs( y0 + getSizeOfDim(1) - y1 ), Math.abs( y1 + getSizeOfDim(1) - y0 ) );
		
		return Math.min(xDirectDist, xNonDirectDist) + Math.min(yDirectDist, yNonDirectDist);
	}
	
	public Collection<GridPos> getNeighbours( GridPos pos ) {
		Collection<GridPos> l = super.getNeighbours(pos);
		
		int x = pos.getPosVector()[0];
		int y = pos.getPosVector()[1];
				
		if( x == 0 )
			l.add( new GridPos( getSizeOfDim(0) - 1, y ) );
		if( x == getSizeOfDim(0) -1 )
			l.add( new GridPos( 0, y ) );
		
		if( y == 0 )
			l.add( new GridPos( x, getSizeOfDim(1) - 1 ) );
		if( y == getSizeOfDim(1) -1 )
			l.add( new GridPos( x, 0 ) );	
		return l;
	}
}

