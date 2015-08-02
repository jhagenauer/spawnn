

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.GridPos;
import spawnn.som.grid.GrowingGrid2D;
import spawnn.som.kernel.ConstantKernel;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class GHSOM {
	
	private static Logger log = Logger.getLogger(GHSOM.class);
	
	private static Random r = new Random();
	private static int lambda = 10000;
	private static double tau_1 = 0.7, tau_2 = 0.4;
	
	class Parent {
		Parent(GridPos p, Grid<double[]> grid ) { this.p = p; this.grid = grid; }
		GridPos p;
		Grid<double[]> grid;
	}

	public static void main(String[] args) {
				
		List<double[]> samples = DataUtils.readCSV("data/iris.csv");
		Dist<double[]> fDist = new EuclideanDist();
				
		Grid2D<double[]> grid = new Grid2D<double[]>(1, 1 );
		SomUtils.initRandom(grid, samples);
					
		BmuGetter<double[]> bg = new DefaultBmuGetter<double[]>(fDist);
		
		SOM som = new SOM( new ConstantKernel(1), new LinearDecay( 1.0, 0.0 ), grid, bg );
		for (int t = 0; t < lambda; t++) {
			double[] x = samples.get(r.nextInt(samples.size()));
			som.train( (double)t/lambda, x );
		}
		
		double mqe_0 = DataUtils.getQuantizationError( grid.getPrototypes().iterator().next(), samples, fDist );
		log.debug("mqe_0: "+mqe_0);
		
		trainHierarchical( samples, bg, fDist, mqe_0, mqe_0 );	
	}
	
	public static void trainHierarchical( List<double[]> samples, BmuGetter<double[]> bg, Dist<double[]> fDist, final double mqe_u, final double mqe_0 ) {
		GrowingGrid2D grid = new GrowingGrid2D(1, 1 );
		SomUtils.initRandom(grid, samples);
		
		while( true ) {
			
			SOM som = new SOM( new GaussKernel( new LinearDecay( Math.max(1, grid.getMaxDist()), 1 ) ), new LinearDecay( 1,0.0 ), grid, bg );
			for (int t = 0; t < lambda; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				som.train( (double)t/lambda, x );
			}
			
			Map<GridPos,Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bg);
			Map<GridPos,Double> qes = new HashMap<GridPos,Double>();
			for( GridPos p : bmus.keySet() )
				qes.put( p, DataUtils.getQuantizationError( grid.getPrototypeAt(p), bmus.get(p), fDist) );
			
			double MQE_m = 0;
			for( double d : qes.values() )
				MQE_m += d;
			MQE_m /= qes.size();
						
			log.debug("1ct: "+MQE_m+" < "+ (tau_1 * mqe_u) );
			if( MQE_m < tau_1 * mqe_u ) 
				break;
			
			 // grow horizontally
			// get pos with largest qe 
			double maxQE = 0;
			GridPos maxP = null;
			for( GridPos p : bmus.keySet() ) 
				if( qes.get(p) > maxQE ) {
					maxQE = qes.get(p);
					maxP = p;
				}
			
			// get neighbor with largest dist
			double maxDistNB = 0;
			GridPos maxNB = null;
			for( GridPos nb : grid.getNeighbours(maxP) ) 
				if( fDist.dist(grid.getPrototypeAt(nb), grid.getPrototypeAt(maxP) ) > maxDistNB ) {
					maxDistNB = fDist.dist(grid.getPrototypeAt(nb), grid.getPrototypeAt(maxP) );
					maxNB = nb;
				}
			
			// insert new row/column
			if( maxNB == null ) {
				grid.addVector(0, 0);
			} else if( maxP.getPosVector()[1] == maxNB.getPosVector()[1] ) { // insert column
				grid.addVector(0, Math.min( maxP.getPosVector()[0], maxNB.getPosVector()[0]) );
			} else { // insert row
				log.error("Inserting row!");
				grid.addVector(1, Math.max( maxP.getPosVector()[1], maxNB.getPosVector()[1]) );
			}			
		}
		
		log.debug("training finished. grid: "+grid);
		
		// check all prototypes and train hierarchical
		Map<GridPos,Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bg);
		for( GridPos p : bmus.keySet() ) {
			double mqe_i = DataUtils.getQuantizationError( grid.getPrototypeAt(p), bmus.get(p), fDist);
			log.debug("2ct: "+mqe_i+" < "+ (tau_2 * mqe_0) );
			if( !( mqe_i < tau_2 * mqe_0 ) )
				trainHierarchical( new ArrayList<double[]>(bmus.get(p)), bg, fDist, mqe_i, mqe_0);
		}
	}
}
