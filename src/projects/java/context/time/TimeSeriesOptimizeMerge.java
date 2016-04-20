package context.time;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetterTimeMSOM;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.kernel.KernelFunction;
import spawnn.som.net.ContextSOM;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class TimeSeriesOptimizeMerge {

	private static Logger log = Logger.getLogger(TimeSeriesOptimizeMerge.class);

	public static void main(String[] args) {
		try {

			final List<double[]> samples = DataUtils.readCSV(new FileInputStream("data/mg/mgsamples.csv") ).subList(0, 150000);

			final Dist<double[]> fDist = new EuclideanDist( new int[] { 1 });
			// Dist tDist = new SubDist(eDist, new int[]{0});
			
			final int T_MAX = samples.size();

			final int rcpFieldSize = 30;

			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (double alpha = 1.0; alpha > 0.0; alpha -= 0.01) {

				final double ALPHA = alpha;

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
						
						SummaryStatistics s = new SummaryStatistics();
						SummaryStatistics e = new SummaryStatistics();
						
						for( int i = 0; i < 4; i++ ) {
							
							/*Grid2D<double[]> grid = new Grid2D<double[]>(10,10);
							Random r = new Random();
							for( GridPos gp : grid.getPositions() ) {
								double[] rs = samples.get( r.nextInt(samples.size() ) );
								double[] d = Arrays.copyOf(rs, rs.length+grid.getNumDimensions() );
								grid.setPrototypeAt(gp, d );
							}
									
							BmuGetterTimeSD bg = new BmuGetterTimeSD(fDist, ALPHA );								
							Som som = new SomSD( new GaussKernel( new LinearDecay(10, 1)), new LinearDecay(1.0,0.0), grid, bg, samples.get(0).length );*/
						
							/*SorterTimeMNG bg = new SorterTimeMNG(fDist, ALPHA, 0.75);
							MNG ng = new MNG(100, 50.0, 0.01, 0.5, 0.005, 4, bg );*/
														
							Grid2D<double[]> grid = new Grid2D<double[]>(10,10);
							Random r = new Random();
							for( GridPos gp : grid.getPositions() ) {
								double[] rs = samples.get( r.nextInt(samples.size() ) );
								double[] d = Arrays.copyOf(rs, rs.length*2 );
								grid.setPrototypeAt(gp, d );
							}
								
							BmuGetterTimeMSOM bg = new BmuGetterTimeMSOM(fDist, ALPHA,0.75);
							KernelFunction nbKernel = new GaussKernel(new LinearDecay(10, 1));
							DecayFunction learningRate = new LinearDecay(1.0, 0.0);
							SOM som = new ContextSOM(nbKernel, learningRate, grid, bg, samples.get(0).length );
	
							for (int t = 0; t < T_MAX; t++) {
	
								double[] x = samples.get(t % samples.size());
								som.train((double) t / T_MAX, x);
							}
	
							bg.setLastBmu(null);
							
							//Map<double[],Set<double[]>> bmus = NgUtils.getBmuMapping(samples, ng.getNeurons(), bg);
							Map<GridPos,Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bg);
							double[] tqe = TimeSeries.getTemporalQuantizationError(samples, bmus, fDist, rcpFieldSize);
	
							double sum = 0;
							for (double d : tqe)
								sum += d;
							s.addValue(sum);
	
							double entr = SomUtils.getEntropy(samples, bmus);
							e.addValue(entr);
	
							
						}
						
						double[] dd = new double[] { s.getMean(), e.getMean(), ALPHA };
						
						log.info(Arrays.toString(dd));

						return dd;
						
					}

				}));

			}

			es.shutdown();

			double[] min = null;

			for (Future<double[]> f : futures) {
				double[] d = f.get();

				if (min == null || d[0] < min[0])
					min = d;

			}
			log.info("Min: " + Arrays.toString(min));

		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

}
