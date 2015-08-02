package spawnn_toolkit;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;

public class TestNetworks {
	private static Logger log = Logger.getLogger(TestNetworks.class);
	
	/* 12x8, 96 neurons
	 * 
	 * 2015-04-10 19:53:17,455 DEBUG [main] spawnn_toolkit.TestNetworks: 3, 12X8:3.0688470319212873,0.17471666297583704
2015-04-10 19:56:17,722 DEBUG [main] spawnn_toolkit.TestNetworks: 4, 12X8:2.3940635456297814,0.0882780868168954
2015-04-10 19:59:21,040 DEBUG [main] spawnn_toolkit.TestNetworks: 5, 12X8:2.077212093700247,0.11068076373587686
2015-04-10 20:02:26,056 DEBUG [main] spawnn_toolkit.TestNetworks: 6, 12X8:1.8571921229407284,0.09201169895370546
2015-04-11 08:15:47,414 DEBUG [main] spawnn_toolkit.TestNetworks: 13, 12X8:1.5455918382028502,0.021280581466311885
2015-04-10 20:06:19,819 DEBUG [main] spawnn_toolkit.TestNetworks: 23:2.9849976183525633,0.074722026884974
2015-04-10 20:10:50,727 DEBUG [main] spawnn_toolkit.TestNetworks: 35:2.3882036315372654,0.08410726931527987
2015-04-10 20:15:50,484 DEBUG [main] spawnn_toolkit.TestNetworks: 44:1.9679738178697699,0.07355623478659071
2015-04-10 20:20:59,709 DEBUG [main] spawnn_toolkit.TestNetworks: 45:1.9105380219629953,0.06353774162770241
2015-04-10 20:26:06,616 DEBUG [main] spawnn_toolkit.TestNetworks: 46:1.8787764284389783,0.06542630872073298
2015-04-10 20:31:17,252 DEBUG [main] spawnn_toolkit.TestNetworks: 47:1.8383434264603318,0.05744143852452829
2015-04-11 08:23:38,256 DEBUG [main] spawnn_toolkit.TestNetworks: 96:1.2350662686617158,0.03505503314550277
	   */

	public static void main(String[] args) {
		final Random r = new Random();
		final int T_MAX = 100000;
		int runs = 200;
		final DataFrame df = DataUtils.readSpatialDataFrameFromShapefile(new File("output/data_philly.shp"), true);

		final int[] ga = new int[] { 1, 2 };
		final int[] fa = new int[] { 4, 5, 6, 7, 8, 9, 10, 11, 13 };

		final Dist<double[]> fDist = new EuclideanDist(fa);
		final Dist<double[]> gDist = new EuclideanDist(ga);

		DataUtils.zScoreColumns(df.samples, fa);
		DataUtils.zScoreGeoColumns(df.samples, ga, gDist );

		for (final int l : new int[] { /*3, 4, 5, 6*/ 13 }) {
			for (final int[] dim : new int[][] { { 12, 8 } }) {

				ExecutorService es = Executors.newFixedThreadPool(4);
				List<Future<Double>> futures = new ArrayList<Future<Double>>();
				for (int i = 0; i < runs; i++) {

					futures.add(es.submit(new Callable<Double>() {

						@Override
						public Double call() throws Exception {
							Grid2D<double[]> grid = new Grid2DHex<>(dim[0], dim[1]);
							SomUtils.initRandom(grid, df.samples);

							BmuGetter<double[]> bg = new KangasBmuGetter<double[]>(gDist, fDist, l);
							SOM som = new SOM(new GaussKernel(new LinearDecay(10, 1)), new LinearDecay(1, 0.0), grid, bg);
							for (int t = 0; t < T_MAX; t++) {
								double[] x = df.samples.get(r.nextInt(df.samples.size()));
								som.train((double) t / T_MAX, x);
							}

							Map<GridPos, Set<double[]>> bmus = SomUtils.getBmuMapping(df.samples, grid, bg);
							Map<double[], Set<double[]>> nBmus = new HashMap<double[], Set<double[]>>();
							for (GridPos p : bmus.keySet())
								nBmus.put(grid.getPrototypeAt(p), bmus.get(p));

							double qe = DataUtils.getMeanQuantizationError(nBmus, fDist);
							double ge = DataUtils.getMeanQuantizationError(nBmus, gDist);
							return qe / ge;
						}
					}));
				}
				
				es.shutdown();

				DescriptiveStatistics ds = new DescriptiveStatistics();
				for (Future<Double> f : futures)
					try {
						ds.addValue(f.get());
					} catch (InterruptedException e) {
						e.printStackTrace();
					} catch (ExecutionException e) {
						e.printStackTrace();
					}
				log.debug(l + ", " + dim[0] + "X" + dim[1] + ":" + ds.getMean() + "," + ds.getStandardDeviation());
			}
		}

		final int nrNeurons = 96;
		for (final int l : new int[] { /*23, 35, 44, 45, 46, 47*/ nrNeurons }) {

			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<Double>> futures = new ArrayList<Future<Double>>();
			for (int i = 0; i < runs; i++) {

				futures.add(es.submit(new Callable<Double>() {

					public Double call() throws Exception {
						Sorter<double[]> s = new KangasSorter<double[]>(gDist, fDist, l);
						NG ng = new NG(nrNeurons, nrNeurons/2, 0.01, 0.5, 0.005, df.samples.get(0).length, s);

						for (int t = 0; t < T_MAX; t++) {
							double[] x = df.samples.get(r.nextInt(df.samples.size()));
							ng.train((double) t / T_MAX, x);
						}

						Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(df.samples, ng.getNeurons(), s);

						double qe = DataUtils.getMeanQuantizationError(bmus, fDist);
						double ge = DataUtils.getMeanQuantizationError(bmus, gDist);
						return qe / ge;
					}
				}));
			}
			
			es.shutdown();

			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (Future<Double> f : futures)
				try {
					ds.addValue(f.get());
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			log.debug(l + ":" + ds.getMean() + "," + ds.getStandardDeviation());

		}
	}
}
