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

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetterTimeSD;
import spawnn.som.decay.ConstantDecay;
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

public class TimeSeriesOptimizeSD {

	private static Logger log = Logger.getLogger(TimeSeriesOptimizeSD.class);

	public static void main(String[] args) {
		try {

			class TimeDist implements Dist<double[]> {
				int tIdx = -1;

				TimeDist(int tIdx) {
					this.tIdx = tIdx;
				}

				@Override
				public double dist(double[] a, double[] b) {
					if (a[tIdx] > b[tIdx])
						return Double.MAX_VALUE;
					else
						return Math.sqrt(Math.pow(a[tIdx] - b[tIdx], 2));
				}
			}

			final List<double[]> samples = DataUtils.readCSV(new FileInputStream("data/mg/mgsamples.csv") ).subList(0, 150000);

			final Dist<double[]> eDist = new EuclideanDist();
			final Dist<double[]> fDist = new EuclideanDist( new int[] { 1 });
			// Dist tDist = new SubDist(eDist, new int[]{0});
			final Dist<double[]> tDist = new TimeDist(0);

			final int T_MAX = 150000;

			final int rcpFieldSize = 30;

			ExecutorService es = Executors.newFixedThreadPool(12);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (double alpha = 1.0; alpha > 0.0; alpha -= 0.01) {

				final double ALPHA = alpha;

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
						Grid2D<double[]> grid = new Grid2D<double[]>(10, 10);

						// init grid
						Random r = new Random();
						for (GridPos gp : grid.getPositions()) {
							double[] rs = samples.get(r.nextInt(samples.size()));
							double[] d = Arrays.copyOf(rs, rs.length + grid.getNumDimensions() );
							grid.setPrototypeAt(gp, d);
						}

						BmuGetterTimeSD bg = new BmuGetterTimeSD(fDist, ALPHA);

						// neighborhood function
						KernelFunction nbKernel = new GaussKernel(new LinearDecay(10, 1));

						// learning rate
						// DecayFunction learningRate = new ExponentialDecay(1.0, 0.0, 0.4, 1.0 );
						DecayFunction learningRate = new LinearDecay(1.0, 0.0);

						// alpha rate
						// DecayFunction alphaRate = new ExponentialDecay(1.0,ALPHA, 0.3, 1.0 );
						//DecayFunction alphaRate = new LinearDecay(1.0, ALPHA);
						DecayFunction alphaRate = new ConstantDecay(ALPHA);

						SOM som = new ContextSOM(nbKernel, learningRate, grid, bg, samples.get(0).length);

						for (int t = 0; t < T_MAX; t++) {

							bg.setAlpha(alphaRate.getValue((double) t / T_MAX));

							double[] x = samples.get(t % samples.size());
							som.train((double) t / T_MAX, x);
						}

						bg.setLastBmu(null);
						Map<GridPos, Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bg);
						double[] tqe = TimeSeries.getTemporalQuantizationError(samples, bmus, fDist, rcpFieldSize);

						double sum = 0;
						for (double d : tqe)
							sum += d;

						double entr = SomUtils.getEntropy(samples, bmus);

						double[] dd = new double[] { sum, ALPHA, entr };

						if (entr > 1)
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
