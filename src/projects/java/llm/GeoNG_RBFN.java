package llm;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
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

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.ng.NG;
import spawnn.ng.ContextNG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.sorter.SorterWMC;
import spawnn.ng.utils.NGUtils;
import spawnn.rbf.RBF;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class GeoNG_RBFN {

	private static Logger log = Logger.getLogger(GeoNG_RBFN.class);

	public static void main(String[] args) {
		final int nrNeurons = 4;
		final int T_MAX = 100000;
		final Random r = new Random();
		
		SpatialDataFrame sd = DataUtils.readSpatialDataFrameFromShapefile(new File("data/marco/kuntz/haeuser_PGOstadtregion.shp"), true);
		
		Set<Integer> attrs = new HashSet<Integer>();
		//for (int i = 5; i < 20; i++) attrs.add(i);
		 
		attrs.add(17);
		attrs.add(18);
		attrs.add(13); 
		attrs.add(6);
				
		List<double[]> all = new ArrayList<double[]>();
		for (double[] d : sd.samples) {
			double[] nd = new double[attrs.size() + 3];
			nd[0] = d[0];
			nd[1] = d[1];

			int j = 2;
			for (int i : attrs)
				nd[j++] = d[i];

			nd[nd.length - 1] = d[4];
			all.add(nd);
		}

		final int[] ga = new int[] { 0, 1 };
		final int[] fa1 = new int[all.get(0).length - 3];
		for (int i = 0; i < fa1.length; i++)
			fa1[i] = i + 2;
		final int fa2 = all.get(0).length - 1;
		
		DataUtils.zScoreGeoColumns(all, ga, new EuclideanDist(ga));
		DataUtils.zScoreColumns(all, fa1);
		DataUtils.zScoreColumn(all, fa2);
				
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa1);
		final int maxK = 10;

		Map<Integer, double[][]> settings = new HashMap<Integer, double[][]>();
		//settings.put(0, new double[][] { { 1, nrNeurons, 1 }, { 0, 0, 1 } }); // cng
		settings.put(0, new double[][] { { nrNeurons, nrNeurons, 1 }, { 0, 0, 1 } }); // cng
		//settings.put(1, new double[][] { { 0.3, 0.3, 0.1 }, { 0, 0, 1 } }); // wng
		//settings.put(2, new double[][] { { 0.2, 0.2, 0.01 }, { 0.1, 0.1, 0.02 } }); // wmng

		for (final int m : settings.keySet()) {
			double[][] s = settings.get(m);
			for (double a = s[0][0]; a <= s[0][1]; a += s[0][2]) {
				for (double b = s[1][0]; b <= s[1][1]; b += s[1][2]) {
					final double A = a, B = b;
					
					ExecutorService es = Executors.newFixedThreadPool(4);
					List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

					for (int j = 0; j < 16; j++) {
						Collections.shuffle(all);
						final List<double[]> samples = new ArrayList<double[]>();
						final List<double[]> desired = new ArrayList<double[]>();

						for (double[] d : all) {

							double[] dd = new double[ga.length+fa1.length];
							for (int i = 0; i < ga.length; i++)
								dd[i] = d[ga[i]];
							for (int i = 0; i < fa1.length; i++)
								dd[i+ga.length] = d[fa1[i]];
							samples.add(dd);
							
							desired.add(new double[] { d[fa2] });
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

									List<double[]> centers = null;
									
									double fqe = 0, gqe = 0;
									if (m == 0) { // CNG
										Sorter<double[]> s = new KangasSorter<double[]>(gDist, fDist, (int) A);
										NG ng = new NG(nrNeurons, nrNeurons / 2, 0.01, 0.5, 0.005, training.get(0).length, s);

										for (int t = 0; t < T_MAX; t++) {
											double[] x = training.get(r.nextInt(training.size()));
											ng.train((double) t / T_MAX, x);
										}
										centers = ng.getNeurons();
										fqe = DataUtils.getMeanQuantizationError(NGUtils.getBmuMapping(training, ng.getNeurons(), s), fDist);
										gqe = DataUtils.getMeanQuantizationError(NGUtils.getBmuMapping(training, ng.getNeurons(), s), gDist); 
										
									} else if (m == 1) { // WNG
										Map<Dist<double[]>, Double> map = new HashMap<Dist<double[]>, Double>();
										map.put(gDist, A);
										map.put(fDist, 1 - A);

										DefaultSorter<double[]> s = new DefaultSorter<double[]>(new WeightedDist<double[]>(map));
										NG ng = new NG(nrNeurons, nrNeurons / 2, 0.01, 0.5, 0.005, training.get(0).length, s);

										for (int t = 0; t < T_MAX; t++) {
											double[] x = training.get(r.nextInt(training.size()));
											ng.train((double) t / T_MAX, x);
										}
										centers = ng.getNeurons();
										fqe = DataUtils.getMeanQuantizationError(NGUtils.getBmuMapping(training, ng.getNeurons(), s), fDist);
										gqe = DataUtils.getMeanQuantizationError(NGUtils.getBmuMapping(training, ng.getNeurons(), s), gDist); 
									} else if (m == 2) { // WMNG
										
										int knn = 8;
										final Map<double[], Map<double[], Double>> dMap = new HashMap<double[], Map<double[], Double>>();
										for (double[] x : training) {
											Map<double[], Double> sub = new HashMap<double[], Double>();
											while (sub.size() <= knn) {

												double[] minD = null;
												for (double[] d : training)
													if (!sub.containsKey(d) && (minD == null || gDist.dist(d, x) < gDist.dist(minD, x)))
														minD = d;
												sub.put(minD, 1.0 / knn);
											}
											dMap.put(x, sub);
										}

										List<double[]> neurons = new ArrayList<double[]>();
										for (int i = 0; i < nrNeurons; i++) {
											double[] rs = training.get(r.nextInt(training.size()));
											double[] d = Arrays.copyOf(rs, rs.length * 2);
											for (int j = rs.length; j < d.length; j++)
												d[j] = r.nextDouble();
											neurons.add(d);
										}

										Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
										for (double[] d : training)
											bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

										SorterWMC s = new SorterWMC(bmuHist, dMap, fDist, A, B);
										ContextNG ng = new ContextNG(neurons, neurons.size() / 2, 0.01, 0.5, 0.005, s);

										s.bmuHistMutable = true;
										for (int t = 0; t < T_MAX; t++) {
											double[] x = training.get(r.nextInt(training.size()));
											ng.train((double) t / T_MAX, x);
										}
										s.bmuHistMutable = false;
										centers = ng.getNeurons();
										fqe = DataUtils.getMeanQuantizationError(NGUtils.getBmuMapping(training, ng.getNeurons(), s), fDist);
										gqe = DataUtils.getMeanQuantizationError(NGUtils.getBmuMapping(training, ng.getNeurons(), s), gDist); 
									}
									
									List<double[]> tmp = new ArrayList<double[]>();
									for( double[]d : centers )
										tmp.add( Arrays.copyOfRange(d, 2, d.length ) );
									centers = tmp;

									EuclideanDist dist = new EuclideanDist();
									Map<double[], Double> hidden = new HashMap<double[], Double>();
									for (double[] c : centers) {
										double d = Double.MAX_VALUE;
										for (double[] n : centers)
											if (c != n)
												d = Math.min(d, dist.dist(c, n)*1.2);
										hidden.put(c, d);
									}

									RBF rbf = new RBF(hidden, 1, dist, 0.05);

									for (int i = 0; i < 100000; i++) {
										int j = r.nextInt(training.size());
										double[] x = training.get(j);
										rbf.train(Arrays.copyOfRange(x, 2, x.length), trainingDesired.get(j));
									}
									
									List<double[]> trainingResponse = new ArrayList<double[]>();
									for (double[] x : training)
										trainingResponse.add(rbf.present(Arrays.copyOfRange(x, 2, x.length)));

									List<double[]> validationResponse = new ArrayList<double[]>();
									for (double[] x : validation)
										validationResponse.add(rbf.present(Arrays.copyOfRange(x, 2, x.length)));

									double[] r = new double[] { 
											fqe,
											gqe,
											Meuse.getRMSE(trainingResponse, trainingDesired), 
											Math.pow(Meuse.getPearson(trainingResponse, trainingDesired), 2),										
											Meuse.getRMSE(validationResponse, validationDesired), 
											Math.pow(Meuse.getPearson(validationResponse, validationDesired), 2) 
											};

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
							System.exit(1);
						}
					}
					
					String desc = m+","+A+","+B;
										
					StringBuffer sb = new StringBuffer();
					for (int i = 0; i < ds.length; i++)
						sb.append(ds[i].getMean() + ",");
					log.debug(desc+","+sb.substring(0, Math.min(sb.length(),500)) );
				}
			}

		}

	}
}
