package spawnn.som.grid;

import java.util.Arrays;

public class GridPos implements Comparable<GridPos> {
	
	private final int[] p;
	
	public GridPos( int ... x ) {
		//this.p = x;
		this.p = Arrays.copyOf(x, x.length); // Pos of GridPos should be immutable
	}
	
	@Deprecated // use getPos()
	public final int[] getPosVector() {
		return p;
	}
	
	public int getPos( int i ) {
		return p[i];
	}
	
	public int length() {
		return p.length;
	}
	
	@Override
	public int hashCode() {
		return Arrays.hashCode(p);
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		
		GridPos other = (GridPos) obj;
		if (!Arrays.equals(p, other.p))
			return false;
		return true;
	}
	
	@Override
	public String toString() {
		return Arrays.toString(p);
	}

	@Override
	public int compareTo(GridPos o) {
		if( o.length() != p.length )
			throw new RuntimeException("PosVectors have different lengths!");
		
		for( int i = 0; i < p.length; i++ ) {
			if( p[i] < o.getPos(i) )
				return -1;
			else if( p[i] > o.getPos(i) )
				return 1;
		}
		return 0;
	}
}
