package spawnn.som.grid;

import java.util.Collection;
import java.util.Map;

public class Grid2DHexToroid<T> extends Grid2DHex<T> {
	
	private int xSize, ySize;
	
	public Grid2DHexToroid( int xSize, int ySize ) {
		super(xSize, ySize);
		if( xSize % 2 == 1 )
			throw new RuntimeException("Grid2DHexToroid must have even size of x-dimension!");
		this.xSize = xSize;
		this.ySize = ySize;
	}
					
	public Grid2DHexToroid(Map<GridPos, T> loadGrid) {
		super(loadGrid);
		this.xSize = getSizeOfDim(0);
		this.ySize = getSizeOfDim(1);
	}

	// in output space
	public int dist( GridPos aPos, GridPos bPos ) {
		int x0 = aPos.getPosVector()[0];
		int y0 = aPos.getPosVector()[1];
		int x1 = bPos.getPosVector()[0];
		int y1 = bPos.getPosVector()[1];
		
		// direct dists
		int xDirectDist = Math.abs( x0-x1 );
		int yDirectDist = Math.abs( y0-y1 );
		
		// toro-dists
		int xNonDirectDist = Math.min( Math.abs( x0 + xSize- x1 ), Math.abs( x1 + xSize - x0 ));
		int yNonDirectDist = Math.min( Math.abs( y0 + ySize - y1 ), Math.abs( y1 + ySize - y0 ) );
						
		int a = getHexDist(xDirectDist, yDirectDist, y1 > y0, x0 % 2 == 0);
		int b = getHexDist(xNonDirectDist, yDirectDist, y1 > y0, x0 % 2 == 0);
		
		int c = getHexDist(xDirectDist, yNonDirectDist, y1 < y0, x0 % 2 == 0);
		int d = getHexDist(xNonDirectDist, yNonDirectDist, y1 < y0, x0 % 2 == 0);
		
		return Math.min(a, Math.min(b, Math.min(c,d)));
	}
		
	public Collection<GridPos> getNeighbours( GridPos pos ) {
		Collection<GridPos> l = getNeighborCandidates(pos);
		int xDim = getSizeOfDim(0);
		int yDim = getSizeOfDim(1);
		
		for( GridPos p : l ) {
			int[] pv = p.getPosVector();
			
			// wrap around x
			if( pv[0] < 0 )
				pv[0] = xDim - 1;
			else if( pv[0] > xDim - 1 )
				pv[0] = 0;
			
			// wrap around y
			if( pv[1] < 0 )
				pv[1] = yDim -1;
			else if( pv[1] > yDim -1 )
				pv[1] = 0;
		}
		return l;
	}
}
