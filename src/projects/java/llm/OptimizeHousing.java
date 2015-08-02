package llm;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;

public class OptimizeHousing {

	private static Logger log = Logger.getLogger(OptimizeHousing.class);

	public static void main(String[] args) {
		final Random r = new Random();
		final int T_MAX = 100000;

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("data/marco/dat4/gwr.csv"), new int[] { 6, 7 }, new int[] {}, true);

		final List<double[]> samples = new ArrayList<double[]>();
		final List<Geometry> geoms = new ArrayList<Geometry>();
		final List<double[]> desired = new ArrayList<double[]>();

		List<String> vars = new ArrayList<String>();
		vars.add("xco");
		vars.add("yco");
		vars.add("lnarea_tot");
		vars.add("lnarea_plo");
		vars.add("attic_dum");
		vars.add("cellar_dum");
		vars.add("cond_house_3");
		vars.add("heat_3");
		vars.add("bath_3");
		vars.add("garage_3");
		vars.add("terr_dum");
		vars.add("age_num");
		vars.add("time_index");
		vars.add("zsp_alq_09");
		vars.add("gem_kauf_i");
		vars.add("gem_abi");
		vars.add("gem_alter_");
		vars.add("ln_gem_dic");

		for (double[] d : sdf.samples) {
			if (d[sdf.names.indexOf("time_index")] < 6)
				continue;
			int idx = sdf.samples.indexOf(d);
			double[] nd = new double[vars.size()];
			for (int i = 0; i < nd.length; i++)
				nd[i] = d[sdf.names.indexOf(vars.get(i))];
			samples.add(nd);
			desired.add(new double[] { d[sdf.names.indexOf("lnp")] });
			geoms.add(sdf.geoms.get(idx));
		}

		final int[] fa = new int[] { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 };
		final int[] ga = new int[] { 0, 1 };
		
		DataUtils.zScoreColumns(samples, fa);

		final Dist<double[]> gDist = new EuclideanDist(new int[] { 0, 1 });
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		// 0:[6.0, 0.1, 0.5, 0.001, 2.0, 0.1, 0.5, 0.001],[0.08641520599570356, 0.9667936308338113]
		double[][] bestParams = new double[2][];
		double bestMean[] = new double[2];
		for (final double a : new double[] { 6, 5, 4, 3 })
			for (final double b : new double[] { 0.1, 0.05, 0.01 })
				for (final double c : new double[] { 1.0, 0.5 })
					for (final double d : new double[] { 0.1, 0.05, 0.001 })
						for (final double aa : new double[] { 6, 3, 2 })
							for (final double bb : new double[] { 0.1, 0.05, 0.01 })
								for (final double cc : new double[] { 1.0, 0.5 })
									for (final double dd : new double[] { 0.1, 0.05, 0.001 })
					{
						if (a <= b || c <= d)
							continue;

						ExecutorService es = Executors.newFixedThreadPool(4);
						List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

						for (int i = 0; i < 16; i++) {
							futures.add(es.submit(new Callable<double[]>() {

								@Override
								public double[] call() throws Exception {
									Sorter<double[]> sorter = new DefaultSorter<double[]>(fDist);
									sorter = new ErrorSorter(samples, desired);
									LLMNG ng = new LLMNG(6, a, b, c, d, 
											aa, bb, cc, dd, sorter, fa, samples.get(0).length, 1);
									((ErrorSorter)sorter).setLLMNG(ng);
									
									for (int t = 0; t < T_MAX; t++) {
										int j = r.nextInt(samples.size());
										ng.train((double) t / T_MAX, samples.get(j), desired.get(j));
									}

									List<double[]> response = new ArrayList<double[]>();
									for (double[] x : samples)
										response.add(ng.present(x));
									
									double rmse = Meuse.getRMSE(response, desired);
									if( Double.isNaN(rmse))
										rmse = Double.MAX_VALUE;
									return new double[] { rmse, Math.pow(Meuse.getPearson(response, desired),2.0) };
								}
							}));
						}
						es.shutdown();

						double[] mean = new double[bestMean.length];
						for (Future<double[]> ff : futures) {
							try {
								double[] ee = ff.get();
								for (int i = 0; i < mean.length; i++)
									mean[i] += ee[i] / futures.size();
							} catch (InterruptedException ex) {
								ex.printStackTrace();
							} catch (ExecutionException ex) {
								ex.printStackTrace();
							}
						}

						for (int i = 0; i < mean.length; i++) {
							if (bestParams[i] == null || mean[i] < bestMean[i]) {
								bestParams[i] = new double[] { a, b, c, d, aa, bb, cc, dd };
								bestMean[i] = mean[i];
								log.debug(i + ":" + Arrays.toString(bestParams[i]) + "," + Arrays.toString(mean));
							}
						}

					}

	}
}
