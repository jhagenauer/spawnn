package spawnn.som.grid;

import java.util.ArrayList;
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
	@Override
	public int dist( GridPos aPos, GridPos bPos ) {
		int x0 = aPos.getPos(0);
		int y0 = aPos.getPos(1);
		int x1 = bPos.getPos(0);
		int y1 = bPos.getPos(1);
		
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
		
	@Override
	public Collection<GridPos> getNeighbours( GridPos pos ) {
		int xDim = getSizeOfDim(0);
		int yDim = getSizeOfDim(1);
		
		Collection<GridPos> l = new ArrayList<GridPos>();
		for( GridPos p : getNeighborCandidates(pos) ) {
			int x = p.getPos(0);
			int y = p.getPos(1);
			
			// wrap around x
			if( x < 0 )
				x = xDim - 1;
			else if( x > xDim - 1 )
				x = 0;
			
			// wrap around y
			if( y < 0 )
				y = yDim -1;
			else if( y > yDim -1 )
				y = 0;
			l.add( new GridPos(x,y) );
		}
		return l;
	}
}
