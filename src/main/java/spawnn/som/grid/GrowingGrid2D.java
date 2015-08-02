package spawnn.som.grid;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import spawnn.som.utils.SomUtils;

public class GrowingGrid2D extends Grid2D<double[]> {
	
	public GrowingGrid2D(int xSize, int ySize) {
		super(xSize, ySize);
	}

	public void addVector(int dim, int pos) {
		int vLength = getPrototypes().iterator().next().length; 
				
		Map<GridPos,double[]> old = new HashMap<GridPos,double[]>();
		for( GridPos p : getPositions() )
			old.put( p, getPrototypeAt(p) );
		
		if( dim == 0 ) { // column/X
			// build new grid
			for( GridPos p : old.keySet() ) {
				int x = p.getPosVector()[0];
				int y = p.getPosVector()[1];
				if( x >= pos )
					setPrototypeAt(new GridPos(x+1,y), old.get(p));
				else
					setPrototypeAt( p, old.get(p));
			}
			// init collumn
			for( int i = 0; i < getSizeOfDim(1); i++ )
				setPrototypeAt( new GridPos(pos,i), new double[] {} );
			
			// fill new column
			for( GridPos p1 : getPositions() ) {
				int x1 = p1.getPosVector()[0];
				if( x1 != pos )
					continue;
				
				int nbs = 0;
				double[] d = new double[vLength];
				for( GridPos p2 : getNeighbours(p1) ) {
					if( p2.getPosVector()[0] != x1 ) {
						nbs++;
						for( int i = 0; i < d.length; i++ )
							d[i] += getPrototypeAt(p2)[i];
					}
				}
				for( int i = 0; i < d.length; i++ )
					d[i] /= nbs;
				setPrototypeAt(p1, d);
			}			
		} else if( dim == 1 ) { // row/Y
			// build new grid
			for( GridPos p : old.keySet() ) {
				int x = p.getPosVector()[0];
				int y = p.getPosVector()[1];
				if( y >= pos )
					setPrototypeAt(new GridPos(x,y+1), old.get(p));
				else
					setPrototypeAt( p, old.get(p));
			}
			// init row
			for( int i = 0; i < getSizeOfDim(0); i++ )
				setPrototypeAt( new GridPos(i,pos), new double[] {} );
			
			// fill new column
			for( GridPos p1 : getPositions() ) {
				int y1 = p1.getPosVector()[1];
				if( y1 != pos )
					continue;
				
				int nbs = 0;
				double[] d = new double[vLength];
				for( GridPos p2 : getNeighbours(p1) ) {
					if( p2.getPosVector()[1] != y1 ) {
						nbs++;
						for( int i = 0; i < d.length; i++ )
							d[i] += getPrototypeAt(p2)[i];
					}
				}
				for( int i = 0; i < d.length; i++ )
					d[i] /= nbs;
				setPrototypeAt(p1, d);
			}
		}
	}
		
	public static void main(String[] args) {
		List<double[]> samples = new ArrayList<double[]>();
		samples.add( new double[]{1} );
		samples.add( new double[]{3} );
		
		GrowingGrid2D grid = new GrowingGrid2D( 2, 2 );
		SomUtils.initRandom(grid, samples);
		System.out.println(grid.toString());
		grid.addVector(0, 1 );
		System.out.println(grid.toString());	
	}
}
