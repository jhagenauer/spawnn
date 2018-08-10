package context.cng;

import spawnn.som.grid.Grid2D_Map;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;

public class AvgNeuronsWithinRadius {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		Grid2D_Map<double[]> grid = new Grid2DHex<double[]>(5, 5 );
		
		System.out.println("Max: "+grid.getMaxDist() );
		
		for( int k = 0; k <= grid.getMaxDist(); k++ ) {
			double sum = 0;
			for(GridPos p : grid.getPositions() ) {
				int within = 0;
				for( GridPos nb : grid.getPositions() ) 
					if( grid.dist(p, nb) <= k)
						within++;
			sum += within;				
			}
			System.out.println("Radius "+k+", avg: "+(sum/grid.size() ) );
		}

	}

}
