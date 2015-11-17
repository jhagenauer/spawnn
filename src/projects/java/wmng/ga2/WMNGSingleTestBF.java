package wmng.ga2;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
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
import spawnn.ng.ContextNG;
import spawnn.ng.sorter.SorterWMC;
import spawnn.ng.utils.NGUtils;
import spawnn.som.grid.Grid2D;
import spawnn.utils.DataUtils;
import context.space.binary_field.SpaceTestDiscrete;
import context.space.binary_field.SpaceTestDiscrete2;

public class WMNGSingleTestBF {

	private static Logger log = Logger.getLogger(WMNGSingleTestBF.class);

	public static void main(String[] args) {
		final Random r = new Random();

		final int nrNeurons = 10; // mehr neuronen erhöhen unterschiede
		final int maxDist = 5;

		final int maxRfSize = (maxDist + 1) * (maxDist + 2) * 2 - (maxDist + 1) * 4 + 1; // 2d, rook
		final int fa = 0;
		final int[] ga = new int[] { 1, 2 };
		final Dist<double[]> fDist = new EuclideanDist(new int[] { 0 });
		
		final int T_MAX = 150000;
		int runs = 1024; 
		int threads = 4;
		final boolean normed = true;

		final List<double[]> samples = DataUtils.readCSV("/home/julian/publications/wmdmng/geographical_systems_v2/data/grid/toroid50x50_1.csv");
		final Map<double[], Map<double[], Double>> dMap = SpaceTestDiscrete.readDistMap(samples, "/home/julian/publications/wmdmng/geographical_systems_v2/data/grid/toroid50x50_1.wtg");
		
		List<double[]> params = new ArrayList<double[]>();
		params.add( new double[]{0.25,0.25} );
		params.add( new double[]{0.5,0.25});
		params.add( new double[]{0.75,0.25} );
		params.add( new double[]{0.25,0.5} );
		params.add( new double[]{0.5,0.5});
		params.add( new double[]{0.75,0.5} );
		params.add( new double[]{0.25,0.75} );
		params.add( new double[]{0.5,0.75});
		params.add( new double[]{0.75,0.75} );
		
		long time = System.currentTimeMillis();
		Map<double[], Set<Result>> results = new HashMap<double[], Set<Result>>();
		try {
												
			for (final double[] param : params) {
				log.debug(Arrays.toString(param));

				ExecutorService es = Executors.newFixedThreadPool(threads);
				List<Future<Result>> futures = new ArrayList<Future<Result>>();

				for (int run = 0; run < runs; run++) {
					final int RUN = run;
					futures.add(es.submit(new Callable<Result>() {

						@Override
						public Result call() {

							List<double[]> neurons = new ArrayList<double[]>();
							for (int i = 0; i < nrNeurons; i++) {
								double[] rs = samples.get(r.nextInt(samples.size()));
								double[] d = Arrays.copyOf( rs, rs.length * 2 );
								for (int j = rs.length; j < d.length; j++)
									d[j] = r.nextDouble();
								neurons.add(d);
							}

							Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
							for (double[] d : samples)
								bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

							SorterWMC bg = new SorterWMC(bmuHist, dMap, fDist, param[0], param[1]);
							ContextNG ng = new ContextNG(neurons, (double) neurons.size() / 2, 0.01, 0.5, 0.01, bg);

							bg.bmuHistMutable = true;
							for (int t = 0; t < T_MAX; t++) {
								double[] x = samples.get(r.nextInt(samples.size()));
								ng.train((double) t / T_MAX, x);
							}
							bg.bmuHistMutable = false;

							Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, neurons, bg);
							Map<double[], Set<Grid2D<Boolean>>> rf = SpaceTestDiscrete.getReceptiveFields(samples, dMap, bmus, maxDist, maxRfSize, ga, fa);
							
							Result r = new Result();
							//r.bmus = bmus;
							r.qe = SpaceTestDiscrete2.getUncertainty(rf, maxDist, normed);
							return r;
						}
					}));}
				es.shutdown();
				results.put(param, new HashSet<Result>());
				for (Future<Result> f : futures)
					results.get(param).add(f.get());
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
		log.debug("took: "+(System.currentTimeMillis()-time)/1000.0+"s");
		
		// calc means
		Map<double[], double[]> means = new HashMap<double[], double[]>();
		for (Entry<double[], Set<Result>> e : results.entrySet()) {
			double[] mean = new double[maxDist+1];
			for (Result re : e.getValue())
				for (int i = 0; i < mean.length; i++)
					mean[i] += re.qe[i] / e.getValue().size();

			means.put(e.getKey(), mean);
		}

		// write means to file
		try {
			FileWriter fw = new FileWriter("output/wmng_bf.csv");
			fw.write("# runs " + runs + ", maxDist " + maxDist + "\n");
			fw.write("alpha,beta");
			for (int i = 0; i <= maxDist; i++)
				fw.write(",dist_" + i);
			fw.write("\n");

			for (Entry<double[], double[]> e : means.entrySet()) {
				fw.write(e.getKey()[0] + ","+e.getKey()[1]);
				for (int i = 0; i <= maxDist; i++)
					fw.write("," + e.getValue()[i]);
				fw.write("\n");
			}
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
