package cng_houseprice;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
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

import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.ContextNG;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.sorter.SorterWMC;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class HousepriceOptimize_CV {

	/*
	 * SUCKT, weil CNG(1) immer am besten ist, weil es nutzlos ist, andere Parameter mit einzurechen, da die so oder so schon im ln-model berücksichtigt sind
	 */
	private static Logger log = Logger.getLogger(HousepriceOptimize_CV.class);

	enum method {
		CNG, WMNG
	};

	public static void main(String[] args) {
		boolean firstWrite = true;
		final Random r = new Random();

		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();

		final SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("output/houseprice_no_ctx.csv"), new int[] { 0, 1 }, new int[] {}, true);
		for (double[] d : sdf.samples) {
			double[] nd = Arrays.copyOf(d, d.length - 1);

			samples.add(nd);
			desired.add(new double[] { d[d.length - 1] });
		}

		final int[] fa = new int[samples.get(0).length - 2]; // omit geo-vars
		for (int i = 0; i < fa.length; i++)
			fa[i] = i + 2;
		final int[] ga = new int[] { 0, 1 };

		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);

		DataUtils.zScoreColumns(samples, fa);
		DataUtils.zScoreColumn(desired, 0);
		/* DataUtils.zScoreGeoColumns(samples, ga, gDist); //not necessary */

		// ------------------------------------------------------------------------

		int t_max = 40000;
		double nbFinal = 0.1;
		double lrInit = 0.6;
		double lrFinal = 0.01;

		Map<method, List<double[]>> params = new HashMap<method, List<double[]>>();
		params.put(method.CNG, new ArrayList<double[]>());
		params.put(method.WMNG, new ArrayList<double[]>());

		for (int nrNeurons : new int[] { 4, 8, 12, 16, 20, 24 }) {
			double nbInit = nrNeurons * 3.0 / 2;

			for (int l = 1; l <= nrNeurons; l++)
				params.get(method.CNG).add(new double[] { t_max, nrNeurons, nbInit, nbFinal, lrInit, lrFinal, l, Double.NaN });

			/*for (double alpha = 0; alpha <= 1; alpha += 0.05)
				for (double beta = 0; beta <= 1; beta += 0.05)
					params.get(method.WMNG).add(new double[] { t_max, nrNeurons, nbInit, nbFinal, lrInit, lrFinal, alpha, beta });*/
		}

		int nrParams = 0;
		for (List<double[]> l : params.values())
			nrParams += l.size();
		log.debug("Nr. params: " + nrParams);

		final Map<double[], Map<double[], Double>> rMap = GeoUtils.getRowNormedMatrix(GeoUtils.listsToWeights(GeoUtils.getKNNs(samples, gDist, 8, false)));

		for (final method m : params.keySet())
			for (final double[] param : params.get(m)) {

				ExecutorService es = Executors.newFixedThreadPool(1);
				List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

				for (int run = 0; run < 16; run++) {

					futures.add(es.submit(new Callable<double[]>() {

						@Override
						public double[] call() throws Exception {

							int t_max = (int) param[0];
							int nrNeurons = (int) param[1];
							double lInit = param[2];
							double lFinal = param[3];
							double lr1Init = param[4];
							double lr1Final = param[5];

							DecayFunction nbRate = new PowerDecay(lInit, lFinal);
							DecayFunction lrRate1 = new PowerDecay(lr1Init, lr1Final);

							NG ng = null;
							Sorter<double[]> sorter = null;
							List<double[]> neurons = null;
							if (m == method.CNG) {
								neurons = new ArrayList<double[]>();
								for (int i = 0; i < nrNeurons; i++) {
									double[] d = samples.get(r.nextInt(samples.size()));
									neurons.add(Arrays.copyOf(d, d.length));
								}

								sorter = new KangasSorter<>(new DefaultSorter<>(gDist), new DefaultSorter<>(fDist), (int) param[6]);
								ng = new NG(neurons, nbRate, lrRate1, sorter);
							} else if (m == method.WMNG) {
								neurons = new ArrayList<double[]>();
								for (int i = 0; i < nrNeurons; i++) {
									double[] rs = samples.get(r.nextInt(samples.size()));
									double[] d = Arrays.copyOf(rs, rs.length * 2);
									for (int j = rs.length; j < d.length; j++)
										d[j] = r.nextDouble();
									neurons.add(d);
								}

								Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
								for (double[] d : samples)
									bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

								sorter = new SorterWMC(bmuHist, rMap, fDist, param[6], param[7]);
								ng = new ContextNG(neurons, nbRate, lrRate1, (SorterWMC) sorter);
							}

							if (sorter instanceof SorterWMC)
								((SorterWMC) sorter).setHistMutable(true);

							for (int t = 0; t < t_max; t++) {
								int idx = r.nextInt(samples.size());
								ng.train((double) t / t_max, samples.get(idx));
							}

							if (sorter instanceof SorterWMC)
								((SorterWMC) sorter).setHistMutable(false);

							DescriptiveStatistics rmseDummy = new DescriptiveStatistics();
							DescriptiveStatistics rmseCluster = new DescriptiveStatistics();
							
							for (int k = 0; k < 25; k++) {
								List<double[]> samplesTrain = new ArrayList<double[]>(samples);
								List<double[]> desiredTrain = new ArrayList<double[]>(desired);
								List<double[]> samplesVal = new ArrayList<double[]>();
								List<double[]> desiredVal = new ArrayList<double[]>();

								//while( samplesVal.size() < 100 ) { // Leave most/one out
								while (samplesVal.size() < samples.size() * 0.7) {
									int idx = r.nextInt(samplesTrain.size());
									samplesVal.add(samplesTrain.remove(idx));
									desiredVal.add(desiredTrain.remove(idx));
								}

								{ // cluster as dummy variable
									Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samplesTrain, neurons, sorter);
									List<double[]> sortedNeurons = new ArrayList<double[]>();
									for (Entry<double[], Set<double[]>> e : bmus.entrySet())
										if (!e.getValue().isEmpty())
											sortedNeurons.add(e.getKey());

									double[] y = new double[desiredTrain.size()];
									for (int i = 0; i < desiredTrain.size(); i++)
										y[i] = desiredTrain.get(i)[0];

									double[][] x = new double[samplesTrain.size()][];
									for (int i = 0; i < samplesTrain.size(); i++) {
										double[] d = samplesTrain.get(i);
										x[i] = getStripped(d, fa);
										int length = x[i].length;
										x[i] = Arrays.copyOf(x[i], length + sortedNeurons.size() - 1);
										sorter.sort(d, neurons);
										int idx = sortedNeurons.indexOf(neurons.get(0));
										if (idx < sortedNeurons.size() - 1) // skip last cluster-row
											x[i][length + idx] = 1;
									}
									try {
										// training
										OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
										ols.setNoIntercept(false);
										ols.newSampleData(y, x);
										double[] beta = ols.estimateRegressionParameters();

										// testing
										List<double[]> responseVal = new ArrayList<double[]>();
										for (int i = 0; i < samplesVal.size(); i++) {
											double[] d = samplesVal.get(i);
											double[] xi = getStripped(d, fa);
											int length = xi.length;
											xi = Arrays.copyOf(xi, length + sortedNeurons.size() - 1);
											sorter.sort(d, neurons);

											int idx = sortedNeurons.indexOf(neurons.get(0));
											if (idx < sortedNeurons.size() - 1) // skip last cluster-row
												xi[length + idx] = 1;

											double p = beta[0]; // intercept at beta[0]
											for (int j = 1; j < beta.length; j++)
												p += beta[j] * xi[j - 1];

											responseVal.add(new double[] { p });
										}
										rmseDummy.addValue(Meuse.getRMSE(responseVal, desiredVal));
									} catch (SingularMatrixException e) {
										log.debug(e.getMessage());
										System.exit(1);
									}
								}

								/*{ // a model per cluster
									List<double[]> responseVal = new ArrayList<double[]>();
									List<double[]> responseDes = new ArrayList<double[]>();
									
									Map<double[],Set<double[]>> bmusTrain = NGUtils.getBmuMapping(samplesTrain, neurons, sorter);
									Map<double[],Set<double[]>> bmusVal = NGUtils.getBmuMapping(samplesVal, neurons, sorter);
									
									for (double[] n : neurons ) {									
										List<double[]> subSamplesTrain = new ArrayList<double[]>(bmusTrain.get(n));
										List<double[]> subDesiredTrain = new ArrayList<double[]>();
										for( double[] d : subSamplesTrain )
											subDesiredTrain.add( desiredTrain.get(samplesTrain.indexOf(d)));

										List<double[]> subSamplesVal = new ArrayList<double[]>(bmusVal.get(n));
										
										List<Integer> toStrip = new ArrayList<Integer>();
										for (int f : fa)
											toStrip.add(f);
										for (int j = 0; j < subSamplesTrain.iterator().next().length; j++) {
											DescriptiveStatistics ds = new DescriptiveStatistics();
											for (double[] d : subSamplesTrain)
												ds.addValue(d[j]);
											if (ds.getVariance() == 0.0)
												toStrip.add(j);
										}
										int[] nfa = new int[toStrip.size()];
										for (int i = 0; i < toStrip.size(); i++)
											nfa[i] = toStrip.get(i);

										double[] y = new double[subDesiredTrain.size()];
										for (int i = 0; i < subDesiredTrain.size(); i++)
											y[i] = subDesiredTrain.get(i)[0];

										double[][] x = new double[subSamplesTrain.size()][];
										for (int i = 0; i < subSamplesTrain.size(); i++) {
											double[] d = subSamplesTrain.get(i);
											x[i] = getStripped(d, nfa);
										}

										try {
											// training
											OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
											ols.setNoIntercept(false);
											ols.newSampleData(y, x);
											double[] beta = ols.estimateRegressionParameters();

											// testing
											for (int i = 0; i < subSamplesVal.size(); i++) {
												double[] d = subSamplesVal.get(i);
												double[] xi = getStripped(d, nfa);

												double p = beta[0]; // intercept at beta[0]
												for (int j = 1; j < beta.length; j++)
													p += beta[j] * xi[j - 1];

												responseVal.add(new double[] { p });
												responseDes.add(desiredVal.get(samplesVal.indexOf(d)));
											}
										} catch (SingularMatrixException e) {
											log.debug(e.getMessage());
											System.exit(1);
										}
									}
									rmseCluster.addValue(Meuse.getRMSE(responseVal, responseDes));
								}*/
							}

							return new double[] { rmseDummy.getMean(),rmseCluster.getMean() };
						}
					}));
				}
				es.shutdown();

				DescriptiveStatistics ds[] = null;
				for (Future<double[]> ff : futures) {
					try {
						double[] ee = ff.get();
						if (ds == null) {
							ds = new DescriptiveStatistics[ee.length];
							for (int i = 0; i < ee.length; i++)
								ds[i] = new DescriptiveStatistics();
						}
						for (int i = 0; i < ee.length; i++)
							ds[i].addValue(ee[i]);
					} catch (InterruptedException ex) {
						ex.printStackTrace();
					} catch (ExecutionException ex) {
						ex.printStackTrace();
					}
				}

				try {
					String fn = "output/resultHousepriceCV.csv";
					if (firstWrite) {
						firstWrite = false;
						Files.write(Paths.get(fn), ("method,nrNeurons,param_0,param_1,rmseDummy,rmseCluster\n").getBytes());
					}
					String s = m + "," + param[1] + "," + param[param.length - 2] + "," + param[param.length - 1];
					for (int i = 0; i < ds.length; i++)
						s += "," + ds[i].getMean();
					s += "\n";
					Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
					System.out.print(s);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
	}

	public static double[] getStripped(double[] d, int[] fa) {
		double[] nd = new double[fa.length];
		for (int i = 0; i < fa.length; i++)
			nd[i] = d[fa[i]];
		return nd;
	}
}
