package context.space.redcap.optim;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.RegionUtils;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.Point;

import context.space.SpaceTest;

public class SpaceOptimGeoSOM {

	private static Logger log = Logger.getLogger(SpaceOptimGeoSOM.class);

	public static void main(String[] args) {
		
		final int T_MAX = 150000;
		final int rcpFieldSize = 80;
		final int runs = 25;
		final int threads = 14;
		final Random r = new Random();

		File file = new File("data/redcap/Election/election2004.shp");
		final List<double[]> samples = DataUtils.readSamplesFromShapeFile(file, new int[] {}, true);
		List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(file);
		final Map<double[], Set<double[]>> ctg = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");
		
		// build dist matrix and add coordinates to samples
		Map<double[], Map<double[], Double>> distMap = new HashMap<double[], Map<double[], Double>>();
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			Point p1 = geoms.get(i).getCentroid();

			distMap.put(d, new HashMap<double[], Double>());
			for (double[] nb : ctg.get(d)) {
				int j = samples.indexOf(nb);

				if (i == j)
					continue;

				Point p2 = geoms.get(j).getCentroid();
				distMap.get(d).put(nb, p1.distance(p2));
			}

			// ugly, but easy
			d[0] = p1.getX();
			d[1] = p1.getY();
						
			// normed coordinates
			d[2] = p1.getX();
			d[3] = p1.getY();
		}
					
		final int[] ga = new int[]{0,1};
		final int[] gaNormed = new int[]{2,3};
		
		final int fa = 7;
		final Dist<double[]> fDist = new EuclideanDist( new int[] { fa });
		final Dist<double[]> gDist = new EuclideanDist( ga );
		
		DataUtils.zScoreColumn(samples, fa);
		DataUtils.zScoreGeoColumns(samples, gaNormed, gDist );
		final Dist<double[]> normedGDist = new EuclideanDist( gaNormed );
			
		// build rcpFieldSize-nns
		final Map<double[], List<double[]>> knns = new HashMap<double[], List<double[]>>();
		for (double[] x : samples) {
			List<double[]> sub = new ArrayList<double[]>();
			while (sub.size() <= rcpFieldSize) { // sub.size() must be larger than cLength!

				double[] minD = null;
				for (double[] d : samples)
					if (!sub.contains(d) && (minD == null || gDist.dist(d, x) < gDist.dist(minD, x)))
						minD = d;
				sub.add(minD);
			}
			knns.put(x, sub);
		}
		log.debug("knn build.");
		
		ExecutorService es = Executors.newFixedThreadPool(threads);
						
		for(int k = 0; k < 9; k++ ) {
	
			final int K = k;
			es.execute(new Runnable() {

				@Override
				public void run() {
					
					double mQe[] = new double[rcpFieldSize];
					
					for( int run = 0; run < runs; run++ ) {
											
						spawnn.som.bmu.BmuGetter<double[]> bg = new spawnn.som.bmu.KangasBmuGetter<double[]>( normedGDist, fDist, K );
						Grid2D<double[]> grid = new Grid2DHex<double[]>( 3,3 );
						SomUtils.initRandom(grid, samples);
								
						SOM som = new SOM( new GaussKernel( new LinearDecay(grid.getMaxDist(), 1)), new LinearDecay(1.0,0.0), grid, bg );
						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get( r.nextInt(samples.size()));
							som.train((double) t / T_MAX, x);
						}

						Map<GridPos, Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bg);
						double[] qe = SpaceTest.getQuantizationErrorOld(samples, bmus, fDist, rcpFieldSize, knns);

						for( int i = 0; i < qe.length; i++ )
							mQe[i] += qe[i]/runs;		
					}

					double ints[] = new double[8];
					int numInts = rcpFieldSize/ints.length;
					for( int i = 0; i < ints.length; i++ )
						for( int j = 0; j < numInts; j++ )
							ints[i] += mQe[i*numInts+j];
										
					log.info("GEOSOM "+K+" MQE: "+Arrays.toString(mQe) +", INTS:"+Arrays.toString(ints) );
				}
			});
		}
		es.shutdown();
	}
}
