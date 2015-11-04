package cng_houseprice;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
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
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class HousepriceOptimize_CV {

	private static Logger log = Logger.getLogger(HousepriceOptimize_CV.class);

	enum method {
		CNG, WMNG
	};

	public static void main(String[] args) {
		boolean firstWrite = true;
		final Random r = new Random();
		
		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();

		final SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("output/houseprice.csv"), new int[] { 0, 1 }, new int[] {}, true);
		for (double[] d : sdf.samples) {
			double[] nd = Arrays.copyOf(d, d.length - 1);

			// jitter
			// nd[0] += 0.02+r.nextDouble()*0.01;
			// nd[1] += 0.02+r.nextDouble()*0.01;

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
		int nrNeurons = 16; // mit sehr vieln neuronen ist tatsächlich ein größerer Radius leicht(!) besser
		double lInit = nrNeurons / 2;
		double lFinal = 0.1;
		double lr1Init = 0.7;
		double lr1Final = 0.01;

		Map<method, List<double[]>> params = new HashMap<method, List<double[]>>();

		params.put(method.CNG, new ArrayList<double[]>());
		for (int l = 1; l <= nrNeurons; l++)
			params.get(method.CNG).add( new double[] { t_max, nrNeurons, lInit, lFinal, lr1Init, lr1Final, l, Double.NaN } );
		
		params.put(method.WMNG, new ArrayList<double[]>());
		for ( double alpha = 0; alpha <= 1; alpha += 0.05 )
			for( double beta = 0; beta <= 1; beta += 0.05 )
				params.get(method.WMNG).add(new double[] { t_max, nrNeurons, lInit, lFinal, lr1Init, lr1Final, alpha, beta });
		/*params.get(method.WMNG).add(new double[] { t_max, nrNeurons, lInit, lFinal, lr1Init, lr1Final, 0.625, 0.45 });*/
		
		int nrParams = 0;
		for( List<double[]> l : params.values() )
			nrParams += l.size();
		log.debug("Nr. params: "+nrParams);
		
		for( int dm : new int[]{ 0, 1, 2, 3, 4, 5, 6, 7, 8 } ) {
		
			log.debug("Building dm...");
			final Map<double[], Map<double[], Double>> rMap;
			switch(dm) {
			case 0:
				 rMap = GeoUtils.listsToWeights(GeoUtils.getKNNs(samples, gDist, 4, false));				 
				 break;
			case 1:
				 rMap = GeoUtils.listsToWeights(GeoUtils.getKNNs(samples, gDist, 8, false));
				 break;
			case 2:
				 rMap = GeoUtils.listsToWeights(GeoUtils.getKNNs(samples, gDist, 12, false));
				 break;
			case 3:
				 rMap = GeoUtils.listsToWeights(GeoUtils.getKNNs(samples, gDist, 16, false));
				 break;
			case 4:
				 rMap = GeoUtils.listsToWeights(GeoUtils.getKNNs(samples, gDist, 20, false));
				 break;
			case 5:
				 rMap = GeoUtils.listsToWeights(GeoUtils.getKNNs(samples, gDist, 24, false));
				 break;
			case 6:
				//rMap = GeoUtils.getInverseDistanceMatrix(samples, gDist, 1);
				rMap = GeoUtils.getKNearestMatrix(GeoUtils.getInverseDistanceMatrix(samples, gDist, 1), 1000);
				break;
			case 7:
				//rMap = GeoUtils.getInverseDistanceMatrix(samples, gDist, 2);
				rMap = GeoUtils.getKNearestMatrix(GeoUtils.getInverseDistanceMatrix(samples, gDist, 2), 1000);
				break;
			case 8:
				//rMap = GeoUtils.getInverseDistanceMatrix(samples, gDist, 2);
				rMap = GeoUtils.getKNearestMatrix(GeoUtils.getInverseDistanceMatrix(samples, gDist, 3), 1000);
				break;
			
			default:
				rMap = null;
				System.exit(1);
			}
						 
			//final Map<double[], Map<double[], Double>> rMap = GeoUtils.getInverseDistanceMatrix(samples, gDist, 2);
			//final Map<double[], Map<double[], Double>> rMap = GeoUtils.getKNearestMatrix(GeoUtils.getInverseDistanceMatrix(samples, gDist, 2), 1000);
			
			GeoUtils.rowNormalizeMatrix(rMap);
			log.debug("done");
				
		for (final method m : params.keySet())
			for (final double[] param : params.get(m)) {

				ExecutorService es = Executors.newFixedThreadPool(4);
				List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

				for (int run = 0; run < 12; run++) {

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
							if( m == method.CNG ) {
								neurons = new ArrayList<double[]>();
								for (int i = 0; i < nrNeurons; i++) {
									double[] d = samples.get(r.nextInt(samples.size()));
									neurons.add(Arrays.copyOf(d, d.length));
								}
	
								sorter = new KangasSorter<>(new DefaultSorter<>(gDist), new DefaultSorter<>(fDist), (int) param[6]);
								ng = new NG(neurons, nbRate, lrRate1, sorter);
							} else if( m == method.WMNG ) {
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

								sorter = new SorterWMC(bmuHist, rMap, fDist, param[6], param[7] );
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
							
							List<double[]> sortedNeurons = new ArrayList<double[]>(neurons);

							DescriptiveStatistics rmseDs = new DescriptiveStatistics();
							DescriptiveStatistics r2Ds = new DescriptiveStatistics();
							for (int k = 0; k < 10; k++) {
								List<double[]> samplesTrain = new ArrayList<double[]>(samples);
								List<double[]> desiredTrain = new ArrayList<double[]>(desired);
								List<double[]> samplesVal = new ArrayList<double[]>();
								List<double[]> desiredVal = new ArrayList<double[]>();

								while (samplesVal.size() < samples.size() * 0.7) {
									int idx = r.nextInt(samplesTrain.size());
									samplesVal.add(samplesTrain.remove(idx));
									desiredVal.add(desiredTrain.remove(idx));
								}

								try {
									double[] y = new double[desiredTrain.size()];
									for (int i = 0; i < desiredTrain.size(); i++)
										y[i] = desiredTrain.get(i)[0];

									double[][] x = new double[samplesTrain.size()][];
									for (int i = 0; i < samplesTrain.size(); i++) {
										double[] d = samplesTrain.get(i);
										x[i] = getStripped(d, fa);
										x[i] = Arrays.copyOf(x[i], x[i].length + 1);
										sorter.sort(d, neurons);
										x[i][x[i].length - 1] = sortedNeurons.indexOf(neurons.get(0));
									}
									
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
										xi = Arrays.copyOf(xi, xi.length + 1);
										sorter.sort(d, neurons);
										xi[xi.length - 1] = sortedNeurons.indexOf(neurons.get(0));

										double p = beta[0]; // intercept at beta[0]
										for (int j = 1; j < beta.length; j++)
											p += beta[j] * xi[j - 1];

										responseVal.add(new double[] { p });
									}
									rmseDs.addValue(Meuse.getRMSE(responseVal, desiredVal));
									r2Ds.addValue(Meuse.getR2(responseVal, desiredVal));

								} catch (SingularMatrixException e) {
									log.debug(e.getMessage());
								}
							}

							return new double[] { rmseDs.getMean(), r2Ds.getMean() };
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
						Files.write(Paths.get(fn), ("dm,method,param_0,param_1,rmse,r2\n").getBytes());
					}
					String s = dm+","+m + ","+ param[param.length-2]+","+param[param.length-1];
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
	}

	public static double[] getStripped(double[] d, int[] fa) {
		double[] nd = new double[fa.length];
		for (int i = 0; i < fa.length; i++)
			nd[i] = d[fa[i]];
		return nd;
	}
}
