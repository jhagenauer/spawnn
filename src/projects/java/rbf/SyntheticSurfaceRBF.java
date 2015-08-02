package rbf;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
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
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.rbf.RBF;
import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;

public class SyntheticSurfaceRBF {

	private static Logger log = Logger.getLogger(SyntheticSurfaceRBF.class);

	public static void main(String[] args) {
		final Random r = new Random();

		DataFrame df = DataUtils.readDataFrameFromCSV(new File("output/syn_surface.csv"), new int[] {}, true);

		Map<String, DescriptiveStatistics[]> results = new HashMap<String, DescriptiveStatistics[]>();

		for (int n : new int[] { 2 })
			for (int nnSize : new int[] { 7 }) {

				EuclideanDist dist01 = new EuclideanDist(new int[] { 0, 1 });
				List<double[]> all = new ArrayList<double[]>();
				for (double[] d : df.samples) {

					List<double[]> nns = new ArrayList<double[]>();
					while (nns.size() < nnSize) {

						double[] nn = null;
						for (double[] d2 : df.samples) {
							if (d == d2 || nns.contains(d2))
								continue;

							if (nn == null || dist01.dist(d, d2) < dist01.dist(nn, d))
								nn = d2;
						}
						nns.add(nn);
					}

					double[] d3 = new double[3 + nns.size()];
					d3[0] = d[0]; // x
					d3[1] = d[1]; // y
					d3[2] = d[n]; // n (target)
					for (int i = 0; i < nns.size(); i++)
						d3[i + 3] = nns.get(i)[n];
					all.add(d3);
				}

				int[] ga = new int[] { 0, 1 };
				DataUtils.zScoreGeoColumns(all, ga, new EuclideanDist(ga));

				int[] fa1 = new int[all.get(0).length - 3];
				for (int i = 0; i < fa1.length; i++)
					fa1[i] = i + 3;
				DataUtils.zScoreColumns(all, fa1);

				int fa2 = 2;
				
				int threads = 4;
				int maxRuns = 4;
				final int nrPrototypes = 4;
				final int maxK = 10;

				final int T_MAX = 50000;

				final Dist<double[]> dist = new EuclideanDist();

				ExecutorService es = Executors.newFixedThreadPool(threads);
				List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

				for (int j = 0; j < maxRuns; j++) {
					Collections.shuffle(all);
					final List<double[]> samples = new ArrayList<double[]>();
					final List<double[]> desired = new ArrayList<double[]>();

					for (double[] d : all) {

						double[] n2 = new double[ga.length + fa1.length];
						for (int i = 0; i < ga.length; i++)
							n2[i] = d[ga[i]];
						for (int i = 0; i < fa1.length; i++)
							n2[i + ga.length] = d[fa1[i]];

						samples.add(n2);
						desired.add(new double[] { d[2] });
					}

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

								// NG
								Sorter<double[]> s;
								s = new DefaultSorter<double[]>(dist);
								// s = new KangasSorter<double[]>(gDist, fDist, radius);
								NG ng = new NG(nrPrototypes, (double) nrPrototypes / 2, 0.01, 0.5, 0.005, samples.get(0).length, s);

								for (int t = 0; t < T_MAX * 4; t++)
									ng.train((double) t / T_MAX, samples.get(r.nextInt(samples.size())));
								Map<double[], Set<double[]>> map = NGUtils.getBmuMapping(samples, ng.getNeurons(), s);

								Map<double[], Double> hidden = new HashMap<double[], Double>();
								// min plus overlap
								for (double[] c : map.keySet()) {
									double d = Double.MAX_VALUE;
									for (double[] n : map.keySet())
										if (c != n)
											d = Math.min(d, dist.dist(c, n)) * 1.1;
									hidden.put(c, d);
								}

								RBF rbf = new RBF(hidden, 1, dist, 0.05);
								for (int i = 0; i < T_MAX; i++) {
									int j = r.nextInt(samples.size());
									rbf.train(samples.get(j), desired.get(j));
								}

								List<double[]> response = new ArrayList<double[]>();
								for (double[] x : validation)
									response.add(rbf.present(x));

								double[] r = new double[] { Meuse.getRMSE(response, validationDesired), Math.pow(Meuse.getPearson(response, validationDesired), 2), rbf.getNeurons().size() };

								return r;
							}
						}));
					}
				}

				es.shutdown();

				DescriptiveStatistics[] ds = null;

				for (Future<double[]> f : futures) {
					try {
						double[] d = f.get();

						if (ds == null) {
							ds = new DescriptiveStatistics[d.length];
							for (int i = 0; i < d.length; i++)
								ds[i] = new DescriptiveStatistics();
						}

						for (int i = 0; i < d.length; i++)
							ds[i].addValue(d[i]);
					} catch (InterruptedException e) {
						e.printStackTrace();
					} catch (ExecutionException e) {
						e.printStackTrace();
					}
				}

				String desc = "";

				/*
				 * String desc = ""; if( mod == 0 ) desc += "Basic"; else desc += "Adaptive";
				 * 
				 * if( nnSize > 0 ) desc += " with NBs";
				 */

				StringBuffer sb = new StringBuffer();
				for (int i = 0; i < ds.length; i++)
					sb.append(ds[i].getMean() + ",");
				log.debug(desc + "," + sb.substring(0, Math.min(sb.length(), 500)));

				results.put(desc, ds);

			}

	}
}
