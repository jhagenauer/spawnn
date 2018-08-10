package spawnn.som.grid;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

public class Grid2D_Array extends Grid<double[]> {
	
	private Set<GridPos> positions = new HashSet<>();
	private double[][][] grid;
	
	public Grid2D_Array( int xSize, int ySize ) { 
		super();
		this.grid = new double[xSize][ySize][];
		for( int i = 0; i < xSize; i++ )
			for( int j = 0; j < ySize; j++ )
				positions.add( new GridPos(i,j) );
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
		return grid.length*grid[0].length;
	}

	@Override
	public double[] getPrototypeAt(GridPos pos) {
		return grid[pos.getPos(0)][pos.getPos(1)];
	}

	@Override
	public double[] setPrototypeAt(GridPos pos, double[] v) {
		double[] old = getPrototypeAt(pos);
		this.grid[pos.getPos(0)][pos.getPos(1)] = v;
		return old;
	}
}
