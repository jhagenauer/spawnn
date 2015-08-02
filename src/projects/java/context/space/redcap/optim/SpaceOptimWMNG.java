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
import java.util.concurrent.TimeUnit;

import moomap.jmetal.metaheuristics.smsemoa.NotifyingBlockingThreadPoolExecutor;

import org.apache.log4j.Logger;

import regionalization.RegionUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.ContextNG;
import spawnn.ng.sorter.SorterWMC;
import spawnn.ng.utils.NGUtils;
import spawnn.utils.DataUtils;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.Point;

import context.space.SpaceTest;

public class SpaceOptimWMNG {

	private static Logger log = Logger.getLogger(SpaceOptimWMNG.class);

	public static void main(String[] args) {
		
		final int T_MAX = 150000;
		final int rcpFieldSize = 80;
		final int runs = 4;
		final int threads = 4;
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
		
		ExecutorService es = new NotifyingBlockingThreadPoolExecutor(threads, threads*2, 60, TimeUnit.MINUTES);
		//ExecutorService es = Executors.newFixedThreadPool(threads);
		
		List<Double> bands = new ArrayList<Double>();
		bands.add(-1.0);
		bands.add(0.0);
		
		for ( double b = 160000.0; b < 4500000.0; b += 50000.0 )
			bands.add(b);

		for ( double b : bands ) {
						
		final double BAND = b;
		final Map<double[], Map<double[], Double>> dMap;
		
		if( b < 0 ) {
			dMap = new HashMap<double[],Map<double[],Double>>();
			for( double[] d : ctg.keySet() ) {
				Map<double[],Double> dists = new HashMap<double[],Double>();
				for( double[] nb : ctg.get(d) )
					if( nb != d )
						dists.put( nb, 1.0 );
				
				dMap.put(d, dists );
			}
		} else {
			dMap = SpaceTest.getDistMatrix(samples, gDist, BAND );
			if( dMap == null ) 
				continue;
		}
					
		for (double alpha = 0.5; alpha < 1.0; alpha += 0.1 ) {
		for( double beta = 0.1; beta < 0.5; beta += 0.1 ) {
		
		//for (double alpha : new double[]{1} ) {
		//for( double beta = 0.05; beta < 1.0; beta += 0.05 ) {
		
			final double ALPHA = alpha, BETA = beta;
			es.execute(new Runnable() {

				@Override
				public void run() {
					
					double mQe[] = new double[rcpFieldSize];
										
					for( int run = 0; run < runs; run++ ) {
											
						List<double[]> neurons = new ArrayList<double[]>();
						for( int i = 0; i < 9; i++ ) {
							double[] rs = samples.get(r.nextInt(samples.size()));
							double[] d = Arrays.copyOf(rs, rs.length * 2 );
							for( int j = rs.length; j < d.length; j++ )
								d[j] = r.nextDouble();
							neurons.add( d );
						}
						
						Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
						for (double[] d : samples)
							bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));
						
						SorterWMC bg = new SorterWMC(bmuHist, dMap, fDist, ALPHA, BETA);
						ContextNG ng = new ContextNG(neurons, (double)neurons.size()/2, 0.01, 0.5, 0.005, bg);
												
						bg.bmuHistMutable = true;
						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get( r.nextInt(samples.size()));
							ng.train((double) t / T_MAX, x);	
						}
						bg.bmuHistMutable = false;

						Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, neurons, bg);
						double[] qe = SpaceTest.getQuantizationErrorOld(samples, bmus, fDist, rcpFieldSize, knns);

						for( int i = 0; i < qe.length; i++ )
							mQe[i] += qe[i]/runs;				
						
					}
					
					double ints[] = new double[8];
					int numInts = rcpFieldSize/ints.length;
					for( int i = 0; i < ints.length; i++ )
						for( int j = 0; j < numInts; j++ )
							ints[i] += mQe[i*numInts+j];
					log.info("WMDMNG "+ALPHA+","+BETA+","+BAND+", MQE: "+Arrays.toString(mQe) +", INTS:"+Arrays.toString(ints) );
				}
			});
		}
		}
		}
		es.shutdown();
	}
}
