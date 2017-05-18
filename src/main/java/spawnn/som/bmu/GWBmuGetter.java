package spawnn.som.bmu;

import java.util.Map;
import java.util.Set;

import spawnn.dist.Dist;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;

public class GWBmuGetter extends BmuGetter<double[]> {
	
	private Dist<double[]> fDist, gDist;
	private GWKernel kernel;
	private Map<double[], Double> bandwidth;
	
	public GWBmuGetter(Dist<double[]> gDist, Dist<double[]> fDist, GWKernel k, Map<double[], Double> bandwidth) {
		this.gDist = gDist;
		this.fDist = fDist;
		this.kernel = k;
		this.bandwidth = bandwidth;
	}

	@Override
	public GridPos getBmuPos(double[] x, Grid<double[]> grid, Set<GridPos> ign) {
		double dist = Double.NaN;
		GridPos bmu = null;
		
		for (GridPos p : grid.getPositions()) {

			if (ign != null && ign.contains(p))
				continue;
			
			double[] v = grid.getPrototypeAt(p);
			double w = GeoUtils.getKernelValue( kernel, gDist.dist(v, x), bandwidth.get(x) );
			double[] est = new double[v.length];
			for( int i = 0; i < v.length; i++ )
				est[i] = w * v[i];
			double d = fDist.dist(est, x);
			
			if( bmu == null || d < dist ) { 
				bmu = p;
				dist = d;
			}
		}
		return bmu;
	}
}
