package rbf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.rbf.IncRBF;
import spawnn.utils.DataUtils;

public class OptimParamsAdaptIncRBF {

	private static Logger log = Logger.getLogger(OptimParamsAdaptIncRBF.class);

	public static void main(String[] args) {
		final int T_MAX = 100000;
		final Random r = new Random();

		List<double[]> origSamples = DataUtils.readCSV("data/meuse.csv", new int[] { 9, 13 });
		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();
		for (double[] d : origSamples) {
			samples.add(new double[] { d[1], d[2] });
			desired.add(new double[] { d[6] });
		}

		final Dist<double[]> dist = new EuclideanDist();

		final int maxK = 20;
		long num = 0;

		double bestRMSE = Double.MAX_VALUE;

		List<double[]> settings = new ArrayList<double[]>();
		settings.add(new double[] { 0.05, 0.1, 0.005, 0.001 });
		settings.add(new double[] { 0.0005, 0.005, 0.01, 0.0001 });
		settings.add(new double[] { 75, 85, 100, 125, 175 });
		settings.add(new double[] { 5000, 4000, 3000, 2000, 1000 });
		settings.add(new double[] { 0.5, 0.4, 0.6 });
		settings.add(new double[] { 0.0005, 0.0001, 0.001, 0.01, 0.1, 0.005 });
		settings.add(new double[] { 0.05, 0.01, 0.1, 0.2, 0.005, 0.001, 0.35, 0.5, 0.25 });

		int sum = 0;
		for( double[] d : settings )
			sum += d.length;
		log.debug("num settings: " +sum );

		List<Integer> list = new ArrayList<Integer>();
		while (list.size() < settings.size())
			list.add(list.size());
		Collections.shuffle(list);
		log.debug(list);
		
		final double[] best = new double[] { 0.05, 0.0005, 100, 5000, 0.5, 0.001, 0.05 };

		for (int l : list) {

			for (double value : settings.get(l)) {
				double tmp = best[l];
				best[l] = value;

				ExecutorService es = Executors.newFixedThreadPool(15);
				List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

				for (int p = 0; p < 25; p++)
					for (int k = 0; k < maxK; k++) {
						final int K = k;

						futures.add(es.submit(new Callable<double[]>() {

							@Override
							public double[] call() throws Exception {
								List<double[]> training = new ArrayList<double[]>();
								List<double[]> trainingDesired = new ArrayList<double[]>();
								List<double[]> validation = new ArrayList<double[]>();
								List<double[]> validationDesired = new ArrayList<double[]>();

								for (int i = 0; i < samples.size(); i++) {
									if (K * samples.size() / maxK <= i && i < (K + 1) * samples.size() / maxK) {
										validation.add(samples.get(i));
										validationDesired.add(desired.get(i));
									} else {
										training.add(samples.get(i));
										trainingDesired.add(desired.get(i));
									}
								}

								Map<double[], Double> hidden = new HashMap<double[], Double>();
								while (hidden.size() < 2) {
									double[] d = samples.get(r.nextInt(samples.size()));
									hidden.put(Arrays.copyOf(d, d.length), 1.0);
								}

								IncRBF irbf = new IncRBF(hidden, best[0], best[1], dist, (int) best[2], best[4], best[5], best[6], 1);

								for (int t = 0; t < T_MAX; t++) {
									int idx = r.nextInt(samples.size());
									irbf.train(samples.get(idx), desired.get(idx));
									
									if( t % (int) best[3] == 0 )
										irbf.insert();
								}

								List<double[]> response = new ArrayList<double[]>();
								for (double[] x : validation)
									response.add(irbf.present(x));
								return new double[] { 
										Meuse.getRMSE(response, validationDesired), 
										Math.pow(Meuse.getPearson(response, validationDesired), 2), 
										irbf.getNeurons().size() 
								};
							}
						}));
					}

				es.shutdown();

				DescriptiveStatistics[] ds = new DescriptiveStatistics[3];
				for (int i = 0; i < ds.length; i++)
					ds[i] = new DescriptiveStatistics();

				for (Future<double[]> f : futures) {
					try {
						double[] d = f.get();
						for (int i = 0; i < ds.length; i++)
							ds[i].addValue(d[i]);
					} catch (InterruptedException e) {
						e.printStackTrace();
					} catch (ExecutionException e) {
						e.printStackTrace();
					}
				}

				double rmse = ds[0].getMean();
				double ns = ds[2].getMean();
				
				if (rmse < bestRMSE && ns <= 14) {
					log.info(rmse + "," + ds[1].getMean()+","+Arrays.toString(best) + "," + ns);
					bestRMSE = rmse;
					tmp = best[l]; // keep this value
				}

				log.debug(rmse + "," + ds[1].getMean()+","+Arrays.toString(best) + "," + ns + "," + (num++));

				best[l] = tmp; // restore

			}
		}

	}
}
