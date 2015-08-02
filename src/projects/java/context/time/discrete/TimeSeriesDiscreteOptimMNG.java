package context.time.discrete;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.log4j.Logger;

import context.time.TimeSeries;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.ContextNG;
import spawnn.ng.sorter.SorterMNG;
import spawnn.ng.utils.NGUtils;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class TimeSeriesDiscreteOptimMNG {

	private static Logger log = Logger.getLogger(TimeSeriesDiscreteOptimMNG.class);

	public static void main(String[] args) {
		try {

			final Dist<double[]> fDist = new EuclideanDist(new int[] { 1 });

			final List<double[]> samples = DataUtils.readCSV(new FileInputStream("data/somsd/binary.csv")).subList(0, 1000000);

			final int T_MAX = samples.size();

			ExecutorService es = Executors.newFixedThreadPool(13);
			
			/*
			 6.8808099999999985, entr: 6.390694068478096, alpha: 0.09, beta: 0.35
			 6.898782000000001, entr: 6.418424199922172, alpha: 0.08, beta: 0.39999999999999997
			 6.9069400000000005, entr: 6.413582061261366, alpha: 0.08, beta: 0.35
			 */
			
			for (double alpha = 0; alpha <= 1.0; alpha += 0.01) {
				for( double beta = 0; beta <= 1.0; beta += 0.05 ) {

				final double ALPHA = alpha, BETA = beta;

				es.execute(new Runnable() {

					@Override
					public void run() {

						SorterMNG bg = new SorterMNG(fDist, ALPHA, BETA);
						ContextNG ng = new ContextNG(100, 100 / 2, 0.01, 0.5, 0.005, 4, bg);

						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(t % samples.size());
							ng.train((double) t / T_MAX, x);
						}

						bg.setLastBmu(null);

						Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
						Map<double[], List<List<double[]>>> bmuSeqs = TimeSeries.getReceptiveFields(samples, bmus, 30);
						Map<double[], List<Double>> rf = TimeSeriesDiscrete.getIntersectReceptiveFields(bmuSeqs, 1);

						// get depth
						double depth = 0;
						for( double[] bmu : bmus.keySet() )	
							depth +=  (double)(bmus.get(bmu).size()* rf.get(bmu).size())/samples.size() ;
						
						log.info(depth + "," + ", entr: " + SomUtils.getEntropy(samples, bmus) + ", alpha: " + ALPHA+", beta: "+BETA);

					}
				});
				}
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}
}
