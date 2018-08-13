package spawnn.som.grid;

public class Grid2D_Array<T> extends Grid2D<T> {
	
	private T[][] grid;
	
	public Grid2D_Array( T[][] grid ) {
		super(grid.length,grid[0].length);
		this.grid = grid;
	}
		
	@SuppressWarnings("unchecked")
	public Grid2D_Array( int xSize, int ySize ) { 
		super(xSize,ySize);
		Object[][] o = new Object[xSize][ySize];
		this.grid = (T[][])o;
	}
		
	@Override
	public T getPrototypeAt(GridPos pos) {
		return grid[pos.getPos(0)][pos.getPos(1)];
	}

	@Override
	public T setPrototypeAt(GridPos pos, T v) {
		T old = getPrototypeAt(pos);
		this.grid[pos.getPos(0)][pos.getPos(1)] = v;
		return old;
	}
}
