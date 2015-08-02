package hstsom;

import spawnn.dist.Dist;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;

//h som var 2, grid dists
public class SGridDist<T> implements Dist<double[]> {
	Grid<T> a,b;
				
	public SGridDist(Grid<T> a, Grid<T> b) {
		this.a = a;
		this.b = b;
		
	}

	@Override
	public double dist(double[] d1, double[] d2) {

		// extract fist (A) grid positions from double vector d1 and d2
		// d1 -> gp1A,gp1B, d2 -> gp2A, gp2B
		int[] pos1A = new int[a.getNumDimensions()];
		int[] pos2A = new int[a.getNumDimensions()];
		for( int i = 0; i < pos1A.length; i++ ) {
			pos1A[i] = (int)d1[i];
			pos2A[i] = (int)d2[i];
		}
		GridPos gp1A = new GridPos(pos1A);
		GridPos gp2A = new GridPos(pos2A);
		
		// extract second (B) grid positions
		int[] pos1B = new int[b.getNumDimensions()];
		int[] pos2B = new int[b.getNumDimensions()];
		for( int i = 0; i < pos1B.length; i++ ) {
			pos1B[i] = (int)d1[pos1A.length+i];
			pos2B[i] = (int)d2[pos2A.length+i];
		}
		GridPos gp1B = new GridPos(pos1B);
		GridPos gp2B = new GridPos(pos2B);
				
		// dist = dist in the maps
		return a.dist(gp1A, gp2A) + b.dist(gp1B, gp2B);
	}
	
}
