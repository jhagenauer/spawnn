package rbf;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
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
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.rbf.RBF;
import spawnn.utils.DataUtils;

public class Meuse {

	private static Logger log = Logger.getLogger(Meuse.class);

	public static void main(String[] args) {
		final int T_MAX = 100000;
		final Random r = new Random();

		List<double[]> origSamples = DataUtils.readCSV("data/meuse.csv", new int[] { 9, 13 });
		// Collections.shuffle(origSamples);

		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();
		for (double[] d : origSamples) {
			samples.add(new double[] { d[1], d[2], d[6] });
			desired.add(new double[] { d[6] });
		}

		final int[] ga = new int[] { 0, 1 };
		final int[] fa = new int[] { 2 };

		final int[] ca = new int[ga.length + fa.length];
		for (int i = 0; i < ga.length; i++)
			ca[i] = ga[i];
		for (int i = 0; i < fa.length; i++)
			ca[ga.length + i] = fa[i];

		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		final Dist<double[]> cDist = new EuclideanDist(ca);

		//DataUtils.writeCSV("output/meuse.csv", samples, new String[] { "x", "y", "z" });
		log.debug("samples: " + samples.size());

		double minFDist = Double.MAX_VALUE, minGDist = Double.MAX_VALUE;
		double maxGDist = 0, maxFDist = 0;
		for (double[] a : samples) {
			for (double[] b : samples) {
				if (a == b)
					continue;

				double fd = fDist.dist(a, b);
				double gd = gDist.dist(a, b);

				minFDist = Math.min(minFDist, fd);
				maxGDist = Math.max(maxGDist, gd);
				minGDist = Math.min(minGDist, gd);
				maxFDist = Math.max(maxFDist, fd);
			}
		}

		log.debug("gDist: " + minGDist + " : " + maxGDist);
		log.debug("fDist: " + minFDist + " : " + maxFDist);

		double from = minGDist / maxFDist;
		double to = maxFDist / minGDist;

		from = 0;
		to = 1000;
		log.debug(from + "->" + to);

		final int maxK = samples.size();
		final int numNeurons = 10;
		int threads = 4;
		int num = 1;

		// CNG RBF
		try {
			FileWriter fw = new FileWriter("output/cng.csv");
			fw.write("l,gqe,fqe,rmse,r2\n");
			
			//for (int l = 1; l <= numNeurons; l++) {
			for (int l : new int[]{ 5, 10 } ) {
				final int L = l;

				ExecutorService es = Executors.newFixedThreadPool(threads);
				List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

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

							Sorter<double[]> bmuGetter = new KangasSorter<double[]>(gDist, fDist, L);
							NG ng = new NG(numNeurons, (double) numNeurons / 2, 0.01, 0.5, 0.001, training.get(0).length, bmuGetter);
							for (int t = 0; t < T_MAX; t++) {
								double[] x = training.get(r.nextInt(training.size()));
								ng.train((double) t / T_MAX, x);
							}

							Map<double[], Double> hidden = new HashMap<double[], Double>();
							Map<double[], Set<double[]>> map = NGUtils.getBmuMapping(training, ng.getNeurons(), bmuGetter);
							for (double[] n : map.keySet()) {
								
								double sigma = 0;
								for (double[] x : map.get(n))
									sigma += Math.pow(gDist.dist(x, n), 2);
								sigma = Math.sqrt(sigma/map.get(n).size());
								
								/*double sigma = 0;
								for (double[] x : map.get(n))
									sigma = Math.min( gDist.dist(x, n),sigma );
								sigma *= 1.1;*/
								
								hidden.put(n, sigma);
							}

							RBF rbf = new RBF(hidden, 1, gDist, 0.001);
							for (int t = 0; t < T_MAX; t++) {
								int i = r.nextInt(training.size());
								rbf.train(training.get(i), trainingDesired.get(i));
							}

							List<double[]> response = new ArrayList<double[]>();
							for (double[] x : validation)
								response.add(rbf.present(x));

							return new double[] { 
									DataUtils.getMeanQuantizationError(map, gDist),
									DataUtils.getMeanQuantizationError(map, fDist),
									getRMSE(response, validationDesired), 
									Math.pow(getPearson(response, validationDesired), 2) 
							};
						}

					}));
				}

				es.shutdown();

				DescriptiveStatistics[] ds = new DescriptiveStatistics[4];
				for (int i = 0; i < ds.length; i++)
					ds[i] = new DescriptiveStatistics();

				for (Future<double[]> f : futures) {
					double[] d = f.get();
					for (int i = 0; i < d.length; i++)
						ds[i].addValue(d[i]);
				}
				StringBuffer sb = new StringBuffer();
				for (int i = 0; i < ds.length; i++)
					sb.append("," + ds[i].getMean());
				String s = l + sb.toString();
				log.debug("cng: " + s);

				fw.write(s + "\n");
			}
			fw.close();
			
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		// Weighted RBF
		try {
			FileWriter fw = new FileWriter("output/weighted.csv");
			fw.write("l,gqe,fqe,rmse,r2\n");
			
			for (int n : new int[] { num  } ) {

				//for (double l = from; l <= to; l += (to - from) / n) {
				for(double l : new double[]{ 14 } ) {
					final double L = l;

					ExecutorService es = Executors.newFixedThreadPool(threads);
					List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

					for (int k = 0; k < maxK; k++) {
						final int K = k;

						futures.add(es.submit(new Callable<double[]>() {
							@Override
							public double[] call() throws Exception {

								List<double[]> training = new ArrayList<double[]>();
								List<double[]> trainingScaled = new ArrayList<double[]>();
								List<double[]> trainingDesired = new ArrayList<double[]>();
								List<double[]> validation = new ArrayList<double[]>();
								List<double[]> validationDesired = new ArrayList<double[]>();

								for (int i = 0; i < samples.size(); i++) {
									if (K * samples.size() / maxK <= i && i < (K + 1) * samples.size() / maxK) {
										validation.add(samples.get(i));
										validationDesired.add(desired.get(i));
									} else {
										double[] d = samples.get(i);
										training.add(d);
										
										double[] scaledD = Arrays.copyOf(d, d.length);
										for (int j = 0; j < fa.length; j++)
											scaledD[fa[j]] *= L;
										trainingScaled.add(scaledD);
										
										trainingDesired.add(desired.get(i));
									}
								}

								Sorter<double[]> bmuGetter = new DefaultSorter<double[]>(cDist);
								NG ng = new NG(numNeurons, numNeurons / 2, 0.01, 0.5, 0.001, training.get(0).length, bmuGetter);
								for (int t = 0; t < T_MAX; t++) {
									double[] x = trainingScaled.get(r.nextInt(trainingScaled.size()));
									ng.train((double) t / T_MAX, x);
								}

								Map<double[], Double> hidden = new HashMap<double[], Double>();
								Map<double[], Set<double[]>> map = NGUtils.getBmuMapping(trainingScaled, ng.getNeurons(), bmuGetter);
								for (double[] n : map.keySet()) {
									double sigma = 0;
									for (double[] x : map.get(n))
										sigma += Math.pow(gDist.dist(x, n), 2);
									sigma = Math.sqrt(sigma/map.get(n).size());

									hidden.put(n, sigma );
								}

								RBF rbf = new RBF(hidden, 1, gDist, 0.001);
								for (int t = 0; t < T_MAX; t++) {
									int i = r.nextInt(training.size());
									rbf.train(training.get(i), trainingDesired.get(i));
								}

								List<double[]> response = new ArrayList<double[]>();
								for (double[] x : validation)
									response.add(rbf.present(x));

								return new double[] { 
										DataUtils.getMeanQuantizationError(map, gDist),
										DataUtils.getMeanQuantizationError(map, fDist),
										getRMSE(response, validationDesired), 
										Math.pow(getPearson(response, validationDesired), 2) 
								};
							}
						}));
					}

					es.shutdown();

					DescriptiveStatistics[] ds = new DescriptiveStatistics[4];
					for (int i = 0; i < ds.length; i++)
						ds[i] = new DescriptiveStatistics();

					for (Future<double[]> f : futures) {
						double[] d = f.get();
						for (int i = 0; i < d.length; i++)
							ds[i].addValue(d[i]);
					}
					StringBuffer sb = new StringBuffer();
					for (int i = 0; i < ds.length; i++)
						sb.append("," + ds[i].getMean());
					String s = l + sb.toString();
					log.debug("weighted: " + s);

					fw.write(s + "\n");
				}

			}
			fw.close();
			
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	// Maybe we should remove this function due to triviality
	public static double getRMSE(List<double[]> response, List<double[]> desired) {
		return Math.sqrt( getMSE(response, desired));
	}
	
	public static double getMSE(List<double[]> response, List<double[]> desired) {
		if (response.size() != desired.size())
			throw new RuntimeException("response.size() != desired.size()");

		double mse = 0;
		for (int i = 0; i < response.size(); i++)
			mse += Math.pow(response.get(i)[0] - desired.get(i)[0], 2);
		return mse / response.size();
	}

	// FIXME seems not correct?!?!
	public static double getR2(List<double[]> response, List<double[]> desired) {
		if (response.size() != desired.size())
			throw new RuntimeException();

		double mean = 0;
		for (double[] d : desired)
			mean += d[0];
		mean /= desired.size();

		double ssRes = 0;
		for (int i = 0; i < response.size(); i++)
			ssRes += Math.pow(desired.get(i)[0] - response.get(i)[0], 2);

		double ssTot = 0;
		for (int i = 0; i < desired.size(); i++)
			ssTot += Math.pow(desired.get(i)[0] - mean, 2);

		return 1.0 - ssRes / ssTot;
	}

	public static double getPearson(List<double[]> response, List<double[]> desired) {
		if (response.size() != desired.size())
			throw new RuntimeException();

		double meanDesired = 0;
		for (double[] d : desired)
			meanDesired += d[0];
		meanDesired /= desired.size();

		double meanResponse = 0;
		for (double[] d : response)
			meanResponse += d[0];
		meanResponse /= response.size();

		double a = 0;
		for (int i = 0; i < response.size(); i++)
			a += (response.get(i)[0] - meanResponse) * (desired.get(i)[0] - meanDesired);

		double b = 0;
		for (int i = 0; i < response.size(); i++)
			b += Math.pow(response.get(i)[0] - meanResponse, 2);
		b = Math.sqrt(b);

		double c = 0;
		for (int i = 0; i < desired.size(); i++)
			c += Math.pow(desired.get(i)[0] - meanDesired, 2);
		c = Math.sqrt(c);
		
		if( b == 0 || c == 0 ) // not sure about if this is ok
			return 0;

		return a / (b * c);
	}
}
