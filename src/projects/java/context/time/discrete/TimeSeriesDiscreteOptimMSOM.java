package context.time.discrete;

import java.io.FileInputStream;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.log4j.Logger;

import context.time.TimeSeries;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetterTimeMSOM;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.ContextSOM;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class TimeSeriesDiscreteOptimMSOM {

	private static Logger log = Logger.getLogger(TimeSeriesDiscreteOptimMSOM.class);

	public static void main(String[] args) {
		
		try {

		final List<double[]> samples = DataUtils.readCSV(new FileInputStream("data/somsd/binary.csv")).subList(0, 1000000);

		final int T_MAX = 150000;

		final Dist<double[]> fDist = new EuclideanDist(new int[] { 0 });

		ExecutorService es = Executors.newFixedThreadPool(4);

		for (double alpha = 0; alpha <= 1.0; alpha += 0.01) {
			for( double beta = 0; beta <= 1.0; beta += 0.05 ) {
			
				final double ALPHA = alpha, BETA = beta ;

				es.execute(new Runnable() {

					@Override
					public void run() {

						Grid2DHex<double[]> grid = new Grid2DHex<double[]>(10, 10);

						Random r = new Random();
						for (GridPos gp : grid.getPositions()) {
							double[] rs = samples.get(r.nextInt(samples.size()));
							double[] d = Arrays.copyOf(rs, rs.length * 2 );
							for( int i = rs.length; i < d.length; i++ )
								d[i] = r.nextDouble();
							grid.setPrototypeAt(gp, d);
						}

						BmuGetterTimeMSOM bg = new BmuGetterTimeMSOM(fDist, ALPHA, BETA );

						// main training
						SOM som = new ContextSOM(new GaussKernel(new LinearDecay(10,1)), new LinearDecay(1.0, 0.0), grid, bg,samples.get(0).length);
						for (int t = 0; t < T_MAX; t++) {

							double[] x = samples.get(t % samples.size());
							som.train((double) t / T_MAX, x);
						}

						bg.setLastBmu(null);

						Map<GridPos, Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bg);
						Map<GridPos, List<List<double[]>>> bmuSeqs = TimeSeries.getReceptiveFields(samples, bmus, 30);
						Map<GridPos, List<Double>> rf = TimeSeriesDiscrete.getIntersectReceptiveFields(bmuSeqs, 1);
						
						// calculate depth
						double depth = 0;
						for (GridPos bmu : rf.keySet())
							depth += (double)(rf.get(bmu).size() * bmus.get(bmu).size() ) / samples.size();
						
						log.info(depth+","+SomUtils.getEntropy(samples, bmus)+", alpha: "+ALPHA+", beta: "+BETA);
					}
				});
			}
		}
		
		es.shutdown();

	} catch( Exception e ) {
		e.printStackTrace();
	}
	}
}
