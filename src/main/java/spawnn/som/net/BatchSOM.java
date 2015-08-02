package spawnn.som.net;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import spawnn.som.bmu.BmuGetter;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.KernelFunction;

public class BatchSOM {
			
	protected Grid<double[]> grid; 
	protected BmuGetter<double[]> bmuGetter;
	protected KernelFunction nb;
	
	
	public BatchSOM( KernelFunction nb, Grid<double[]> grid, BmuGetter<double[]> bmuGetter ) {
		this.grid = grid;
		this.bmuGetter = bmuGetter;
		this.nb = nb;
	}
				
	public void train( double t, List<double[]> l) {
		Map<GridPos,List<double[]>> bmus = new HashMap<GridPos,List<double[]>>();
						
		for( double[] x : l ) {
			GridPos p = bmuGetter.getBmuPos( x, grid );
			if( !bmus.containsKey(p) )
				bmus.put(p, new ArrayList<double[]>() );
			bmus.get(p).add(x);
		}
		
		Map<GridPos,double[]> avgMap = new HashMap<GridPos,double[]>();
		for( GridPos p : bmus.keySet() ) {
			double[] avg = new double[l.get(0).length];	
			for( double[] d : bmus.get(p) )
				for( int i = 0; i < avg.length; i++ )
					avg[i] += d[i]/bmus.get(p).size();
			avgMap.put(p,avg);
		}
						
		for( GridPos p : grid.getPositions() ) {
						
			double[] v = new double[l.get(0).length];
			double[] a = new double[v.length]; 
			double[] b = new double[v.length];
			for( GridPos gp : bmus.keySet() ) {		
				double f = nb.getValue( grid.dist(p, gp), t) * bmus.get(gp).size();
				
				for( int j = 0; j < v.length; j++ ) {
					a[j] += f * avgMap.get(gp)[j];
					b[j] += f;
				}
			}					
			for( int j = 0; j < v.length; j++ ) 
				v[j] = a[j]/b[j];
		
			
			grid.setPrototypeAt( p, v );
		}
	}
}
