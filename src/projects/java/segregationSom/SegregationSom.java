package segregationSom;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D_Map;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.kernel.KernelFunction;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;

public class SegregationSom {
	
	private static Logger log = Logger.getLogger(SegregationSom.class);

	public static void main(String[] args) {
		int xDim = 4*7, yDim = 3*7;
		Random r = new Random(0);
		
		int t_max = 100000;
		Grid2D_Map<double[]> grid = new Grid2D_Map<double[]>(xDim, yDim);
		for( int x = 0; x < xDim; x++ )
			for( int y = 0; y < yDim; y++ )
				grid.setPrototypeAt( new GridPos(x,y), new double[]{ r.nextInt(2) } );
		List<GridPos> gps = new ArrayList<>(grid.getPositions());
		
		int nrZeros = 0;
		for( double[] d : grid.getPrototypes() )
			if( d[0] == 0 )
				nrZeros++;
		log.debug("Nr. zeros: "+nrZeros);
		
		SomUtils.printComponentPlane(grid, 0, "output/grid_init.png");

		KernelFunction nb = new GaussKernel(new LinearDecay(0.5, 0.5));
		DecayFunction lr = new LinearDecay(1.0, 0.0);

		for (int t = 0; t < t_max; t++) {
			
			List<double[]> protos = new ArrayList<>(grid.getPrototypes());
			Collections.sort(protos, new Comparator<double[]>() {
				@Override
				public int compare(double[] o1, double[] o2) {
					return Double.compare(o1[0], o2[0]);
				}
			});				
			GridPos bmuPos = gps.get(r.nextInt(gps.size()));
			double[] x = new double[]{ grid.getPrototypeAt(bmuPos)[0] <= protos.get(nrZeros-1)[0] ? 0 : 1 };
			
			double tt = (double) t / t_max;
			for( GridPos p : grid.getPositions() ) {
				double theta = nb.getValue( grid.dist( bmuPos, p ), tt );
				double alpha = lr.getValue( tt );
									
				double[] v = grid.getPrototypeAt(p);
				for( int j = 0; j < v.length; j++ )
					v[j] = v[j] + theta * alpha * (x[j] - v[j]);  
				grid.setPrototypeAt( p, v );
			}
			
		}
		SomUtils.printComponentPlane(grid, 0, "output/grid_final.png");
	}
}
