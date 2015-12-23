package wmng.llm;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.zip.GZIPInputStream;

import llm.LLMNG;

import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.sorter.SorterWMC;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.utils.SomUtils;
import spawnn.utils.GeoUtils;

public class DoubleGridTest {

	private static Logger log = Logger.getLogger(DoubleGridTest.class);
	
	/*TODO 
	 * - modify spBuild (noise), more non-linear, * instead of +, etc
	 * - rerun with 18 neurons?
	**/

	enum model {
		WMNG, LM, NG_LAG, NG
	};

	public static void main(String[] args) {
		long timeAll = System.currentTimeMillis();

		final Random r = new Random();
		
		//int[] nr = new int[]{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48 };
		//int[] nr = new int[]{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
		int[] nr = new int[]{ 16 }; 
		
		int maxRun = 16;//72; 
		int threads = 4;
		
		final int[] fa = new int[] { 2 };
		final Dist<double[]> fDist = new EuclideanDist(fa);

		class GridData {
			Map<double[], Map<double[], Double>> dMapTrain, dMapVal;
			List<double[]> samplesTrain = new ArrayList<double[]>();
			List<double[]> desiredTrain = new ArrayList<double[]>();
			List<double[]> samplesVal = new ArrayList<double[]>();
			List<double[]> desiredVal = new ArrayList<double[]>();

			public GridData(Grid2D<double[]> gridTrain, Grid2D<double[]> gridVal) {
				dMapTrain = GeoUtils.getRowNormedMatrix(GeoUtils.listsToWeights(GeoUtils.getNeighborsFromGrid(gridTrain)));
				for (double[] d : gridTrain.getPrototypes()) {
					samplesTrain.add(d);
					desiredTrain.add(new double[] { d[3] });
				}
				
				dMapVal = GeoUtils.getRowNormedMatrix(GeoUtils.listsToWeights(GeoUtils.getNeighborsFromGrid(gridVal)));
				for (double[] d : gridVal.getPrototypes()) {
					samplesVal.add(d);
					desiredVal.add(new double[] { d[d.length - 1] });
				}
			}
		}
		
		String fn = "output/resultDoubleTest_" + maxRun + "_"+nr.length+".csv";
		try {
			Files.write(Paths.get(fn), ("t_max,nrNeurons,nbInit,nbFinal,lr1Init,lr1Final,lr2Init,lr2Final,model,alpha,beta,rmse,nrmse\n").getBytes());
		} catch (IOException e) {
			e.printStackTrace();
		}

		// load grid data
		final List<GridData> gridData = new ArrayList<GridData>();
		Grid2D<double[]> gridTrain = null;
		for (File fileEntry : new File("/home/julian/publications/geollm/data/grid").listFiles(new FilenameFilter() {
		    public boolean accept(File dir, String name) {
		        return name.toLowerCase().endsWith(".gz");
		    }})) {
			if (fileEntry.isFile()) {
				GZIPInputStream gzis;
				try {
					gzis = new GZIPInputStream(new FileInputStream(fileEntry));
					if( gridTrain == null )
						gridTrain = SomUtils.loadGrid(gzis);
					else {
						gridData.add(new GridData(gridTrain, SomUtils.loadGrid(gzis)));
						gridTrain = null;
					}
					gzis.close();					
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			if( gridData.size() == maxRun )
				break;
		}
		log.debug("GridData: "+gridData.size());

		for( final int T_MAX : new int[]{ 120000 } )
		for( final int nrNeurons : nr ) // je mehr neuronen, desto größer der unterschied?		
		for( final double nbInit : new double[]{ (double)nrNeurons*2.0/3.0 })
		for( final double nbFinal : new double[]{ 1.0 })
		for( final double lr1Init : new double[]{ 0.6 })
		for( final double lr1Final : new double[]{ 0.01 })
		for( final double lr2Init : new double[]{ 0.6 })
		for (final double lr2Final : new double[] { 0.01 }) {
			List<Object[]> models = new ArrayList<Object[]>();
			//models.add(new Object[] { model.LM, null, null });
			models.add(new Object[] { model.NG_LAG, null, null });
			//models.add(new Object[] { model.NG, null, null });
			
			for (double alpha = 0; alpha <= 1; alpha = (double)Math.round( (alpha+0.05) * 100000) / 100000 )
			for (double beta = 0; beta <= 1; beta = (double)Math.round( (beta+0.05) * 100000) / 100000 )
				models.add(new Object[] { model.WMNG, alpha, beta });
			
			//models.add(new Object[] { model.WMNG, 0.65, 0.65 }); // best 16n
			log.debug("models: "+models.size());
						
			for (final Object[] m : models) {

				long time = System.currentTimeMillis();
				ExecutorService es = Executors.newFixedThreadPool(threads);
				Map<GridData, Future<List<double[]>>> futures = new HashMap<GridData, Future<List<double[]>>>();

				for (final GridData data : gridData) {

					futures.put(data, es.submit(new Callable<List<double[]>>() {

						@Override
						public List<double[]> call() throws Exception {

							List<double[]> samplesTrain = data.samplesTrain;
							List<double[]> desiredTrain = data.desiredTrain;
							
							List<double[]> samplesVal = data.samplesVal;
							List<double[]> desiredVal = data.desiredVal;
							Map<double[], Map<double[], Double>> dMapVal = data.dMapVal;
							
							DecayFunction nbRate = new PowerDecay(nbInit, nbFinal);
							DecayFunction lrRate1 = new PowerDecay(lr1Init, lr1Final);
							DecayFunction lrRate2 = new PowerDecay(lr2Init, lr2Final);

							if (m[0] == model.WMNG) { // WMNG + LLM
								double alpha = (double) m[1];
								double beta = (double) m[2];

								List<double[]> neurons = new ArrayList<double[]>();
								for (int i = 0; i < nrNeurons; i++) {
									double[] rs = samplesTrain.get(r.nextInt(samplesTrain.size()));
									double[] d = Arrays.copyOf(rs, rs.length * 2);
									for (int j = rs.length; j < d.length; j++)
										d[j] = r.nextDouble();
									neurons.add(d);
								}

								Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
								for (double[] d : samplesTrain)
									bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

								Map<double[], Map<double[], Double>> dMapTrain = data.dMapTrain;
								SorterWMC sorter = new SorterWMC(bmuHist, dMapTrain, fDist, alpha, beta);

								ContextNG_LLM ng = new ContextNG_LLM(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, fa, 1);
								ng.useCtx = true;
								for (int t = 0; t < T_MAX; t++) {
									int j = r.nextInt(samplesTrain.size());
									ng.train((double) t / T_MAX, samplesTrain.get(j), desiredTrain.get(j));
								}

								sorter.setWeightMatrix(dMapVal); // new weight-matrix

								bmuHist.clear();
								for (double[] d : samplesVal)
									bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

								// train histMap
								for (int i = 0; i < 100; i++) {
									List<double[]> rSamplesVal = new ArrayList<double[]>(samplesVal);
									Collections.shuffle(rSamplesVal);
									for (double[] x : rSamplesVal)
										sorter.sort(x, neurons);
								}

								List<double[]> responseVal = new ArrayList<double[]>();
								for (double[] x : samplesVal)
									responseVal.add(ng.present(x));

								return responseVal;
							} else if( m[0] == model.LM ){ // Linear model

								double[] y = new double[desiredTrain.size()];
								for (int i = 0; i < desiredTrain.size(); i++)
									y[i] = desiredTrain.get(i)[0];

								double[][] x = new double[samplesTrain.size()][];
								for (int i = 0; i < samplesTrain.size(); i++)
									x[i] = getStripped(samplesTrain.get(i), fa);
								try {
									// training
									OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
									ols.setNoIntercept(false);
									ols.newSampleData(y, x);
									double[] beta = ols.estimateRegressionParameters();

									// testing
									List<double[]> responseVal = new ArrayList<double[]>();
									for (int i = 0; i < samplesVal.size(); i++) {
										double[] xi = getStripped(samplesVal.get(i), fa);

										double p = beta[0]; // intercept at beta[0]
										for (int j = 1; j < beta.length; j++)
											p += beta[j] * xi[j - 1];

										responseVal.add(new double[] { p });
									}
									return responseVal;
								} catch (SingularMatrixException e) {
									log.debug(e.getMessage());
									System.exit(1);
								}
								return null;
							} else if( m[0] == model.NG_LAG ){
								List<double[]> lagedSamplesTrain = GeoUtils.getLagedSamples(samplesTrain, data.dMapTrain);
								
								List<double[]> neurons = new ArrayList<double[]>();
								for (int i = 0; i < nrNeurons; i++) {
									double[] d = lagedSamplesTrain.get(r.nextInt(lagedSamplesTrain.size()));
									neurons.add(Arrays.copyOf(d, d.length));
								}
								
								int[] nfa = new int[fa.length*2];
								for( int i = 0; i < fa.length; i++ ) {
									nfa[i] = fa[i];
									nfa[i+fa.length] = fa[i]+samplesTrain.get(0).length;
								}
															
								Sorter<double[]> sorter = new DefaultSorter<>( new EuclideanDist(nfa));

								LLMNG ng = new LLMNG(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, nfa, 1);

								for (int t = 0; t < T_MAX; t++) {
									int j = r.nextInt(lagedSamplesTrain.size());
									ng.train((double) t / T_MAX, lagedSamplesTrain.get(j), desiredTrain.get(j));
								}

								List<double[]> lagedSamplesVal = GeoUtils.getLagedSamples(samplesVal, dMapVal);
								List<double[]> responseVal = new ArrayList<double[]>();
								for ( double[] x : lagedSamplesVal )
									responseVal.add(ng.present(x));
								return responseVal;
								
							} else {
							List<double[]> neurons = new ArrayList<double[]>();
								for (int i = 0; i < nrNeurons; i++) {
									double[] d = samplesTrain.get(r.nextInt(samplesTrain.size()));
									neurons.add(Arrays.copyOf(d, d.length));
								}
														
								Sorter<double[]> sorter = new DefaultSorter<>( new EuclideanDist(fa));
								LLMNG ng = new LLMNG(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, fa, 1);

								for (int t = 0; t < T_MAX; t++) {
									int j = r.nextInt(samplesTrain.size());
									ng.train((double) t / T_MAX, samplesTrain.get(j), desiredTrain.get(j));
								}

								List<double[]> responseVal = new ArrayList<double[]>();
								for ( double[] x : samplesVal )
									responseVal.add(ng.present(x));
								return responseVal;
								
							}
						}
					}));
				}
				es.shutdown();

				// get statistics
				try {
					DescriptiveStatistics ds[] = null;
					for (Entry<GridData, Future<List<double[]>>> ff : futures.entrySet()) {
						List<double[]> responseVal = ff.getValue().get();
						List<double[]> samplesVal = ff.getKey().samplesVal;
						List<double[]> desiredVal = ff.getKey().desiredVal;

						// moran
						/*Map<double[], Map<double[], Double>> dMap = ff.getKey().dMapVal;
						Map<double[], Double> values = new HashMap<double[], Double>();
						for (int i = 0; i < samplesVal.size(); i++)
							values.put(samplesVal.get(i), responseVal.get(i)[0] - desiredVal.get(i)[0]);
						double[] moran = GeoUtils.getMoransIStatistics(dMap, values);*/
						//double[] moran = GeoUtils.getMoransIStatisticsMonteCarlo(dMap, values, 999);
						
						DescriptiveStatistics ds2 = new DescriptiveStatistics();
						for( double[] d : desiredVal )
							ds2.addValue(d[0]);

						double rmse = Meuse.getRMSE(responseVal, desiredVal);
						double[] ee = new double[] { rmse, rmse/ds2.getStandardDeviation() };

						if (ds == null) {
							ds = new DescriptiveStatistics[ee.length];
							for (int i = 0; i < ee.length; i++)
								ds[i] = new DescriptiveStatistics();
						}
						for (int i = 0; i < ee.length; i++)
							ds[i].addValue(ee[i]);

					}

					// write statistics
					String s = T_MAX + "," + nrNeurons + "," + nbInit + "," + nbFinal + "," + lr1Init + "," + lr1Final + "," + lr2Init + "," + lr2Final + "," + Arrays.toString(m).replaceAll("\\[", "").replaceAll("\\]", "");
					for (int i = 0; i < ds.length; i++)
						s += "," + ds[i].getMean();
					s += "\n";
					Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
					System.out.print(s);
				} catch (IOException e) {
					e.printStackTrace();
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
				log.debug("model took: " + (System.currentTimeMillis() - time) / 1000.0 + " sec");
			}
		}
		log.debug("took: " + (System.currentTimeMillis() - timeAll) / 1000.0 / 60.0 + " min");
	}

	public static double[] getStripped(double[] d, int[] fa) {
		double[] nd = new double[fa.length];
		for (int i = 0; i < fa.length; i++)
			nd[i] = d[fa[i]];
		return nd;
	}
}
