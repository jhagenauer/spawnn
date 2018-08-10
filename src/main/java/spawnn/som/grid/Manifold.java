package spawnn.som.grid;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Manifold extends Grid_Map<double[]> {

	Map<GridPos, Set<GridPos>> nbs;
	private List<GridPos> gps;
	private Map<GridPos,Integer> gpsIdxMap;
	
	
	private int[][] dist;

	public Manifold( Map<GridPos,double[]> init, Map<GridPos, Set<GridPos>> nbs ) {
		super(); // simply assume 1 dim
		
		this.nbs = nbs;
		
		// init Grid with samples
		for( GridPos gp : init.keySet() )
			setPrototypeAt(gp, init.get(gp) );
		
		// build distMap
		this.gps = new ArrayList<GridPos>(init.keySet() ); // fixed order list of gps

		this.gpsIdxMap = new HashMap<GridPos,Integer>(); // map gps to idx of ordered list
		for( int i = 0; i < gps.size(); i++ ) 
			gpsIdxMap.put( gps.get(i), i);
				
		this.dist = new int[gps.size()][gps.size()];
		for( int i = 0; i < dist.length; i++ ) {
			for( int j = i; j < dist.length; j++ ) {				
				if( i == j )
					dist[i][j] = 0;
				else { 
					GridPos a = gps.get(i);
					GridPos b = gps.get(j);
					
					if( nbs.containsKey(a) && nbs.get( a ).contains( b ) )
						dist[i][j] = 1; // direct neigbors
					else
						dist[i][j] = 99999;
				}
			}
		}
				
		for( int k = 0; k < dist.length; k++ ) {
			for( int i = 0; i < dist.length; i++ ) {		
				for( int j = i; j < dist.length; j++ ) {
					if( k < i ) {
						if( dist[k][j] + dist[k][i] < dist[i][j] ) 
							dist[i][j] = dist[k][j] + dist[k][i];
									
					} else if( k < j ) { // i <= k
						if( dist[i][k] + dist[k][j] < dist[i][j] ) 
							dist[i][j] = dist[i][k] + dist[k][j];
						
					} else if( k > j ) { 
						if( dist[i][k] + dist[j][k] < dist[i][j] ) 
							dist[i][j] = dist[i][k] + dist[j][k];
						
					}
				}		
			}
		}
	
	}
		
	@Override
	public int dist(GridPos from, GridPos to) {
		int idxFrom = gpsIdxMap.get(from);
		int idxTo = gpsIdxMap.get(to);
		
		if( idxFrom < idxTo )
			return dist[idxFrom][idxTo];
		else
			return dist[idxTo][idxFrom];
	}
	 	
	@Override
	public Collection<GridPos> getNeighbours(GridPos pos) {	
		List<GridPos> nbs = new ArrayList<GridPos>();
		
		for( GridPos p : getPositions() )
			if( dist(p, pos) == 1)
				nbs.add(p);
				
		return nbs; 
	}
}
