package context.time;

import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.ContextNG;
import spawnn.ng.sorter.SorterMNG;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.DataUtils;

public class TimeSeriesMNGOptimize {

	private static Logger log = Logger.getLogger(TimeSeriesMNGOptimize.class);

	public static void main(String[] args) {
		final Random r = new Random();
		
		try {

			final Dist<double[]> fDist = new EuclideanDist(new int[] { 1 });
			// Dist tDist = new SubDist(eDist, new int[]{0});

			final List<double[]> samples = DataUtils.readCSV( new FileInputStream("data/mg/mgsamples.csv")).subList(0, 150000);
			final int T_MAX = samples.size();
			final int rcpFieldSize = 30;

			double[] best = null;
			
			for (double alpha = 0.05; alpha < 1; alpha = (double)Math.round( (alpha+0.05) * 100000) / 100000 ) 
				for (double beta = 0.05; beta < 1; beta = (double)Math.round( (beta+0.05) * 100000) / 100000 ) {

				final double ALPHA = alpha;
				final double BETA = beta;
				
				ExecutorService es = Executors.newFixedThreadPool(4);
				List<Future<Double>> futures = new ArrayList<Future<Double>>();

				for (int i = 0; i < 4; i++) {
				futures.add(es.submit(new Callable<Double>() {

					@Override
					public Double call() throws Exception {


							List<double[]> neurons = new ArrayList<double[]>();
							while( neurons.size() < 100 ){
								double[] rs = samples.get( r.nextInt(samples.size() ) );
								neurons.add( Arrays.copyOf(rs, rs.length*2 ) );
							}
							
							SorterMNG bg = new SorterMNG(fDist, ALPHA, BETA);
							bg.setLastBmu(neurons.get(0));
							
							DecayFunction nbRate = new PowerDecay(50.0, 0.01);
							DecayFunction lrRate = new PowerDecay(0.5, 0.005);
							ContextNG ng = new ContextNG(neurons, nbRate, lrRate, bg );

							for (int t = 0; t < T_MAX; t++) {
								double[] x = samples.get( t % samples.size() );
								ng.train((double) t / T_MAX, x);
							}

							Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
							double[] tqe = TimeSeries.getTemporalQuantizationError(samples,	bmus, fDist, rcpFieldSize);

							double sum = 0;
							for (double d : tqe)
								sum += d;
							return sum;
					}
				}));
				}
				es.shutdown();
				
				DescriptiveStatistics ds = new DescriptiveStatistics();
				for( Future<Double> f : futures ) 
					ds.addValue( f.get() );
				log.debug(ALPHA+","+BETA+","+ds.getMean());
				
				if( best == null || ds.getMean() < best[0] ) { 
					best = new double[]{ds.getMean(),ALPHA,BETA};
					log.debug("best: "+Arrays.toString(best));
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
