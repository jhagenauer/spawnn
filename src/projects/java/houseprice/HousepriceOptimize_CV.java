package houseprice;

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
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import llm.LLMNG;

import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.rbf.RBF;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.transform;
import spawnn.utils.SpatialDataFrame;

public class HousepriceOptimize_CV {

	private static Logger log = Logger.getLogger(HousepriceOptimize_CV.class);

	public static void main(String[] args) {
		boolean firstWrite = true;
		final Random r = new Random();

		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();
		final SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("output/houseprice.csv"), new int[] { 0,1 }, new int[] {}, true);
		for (double[] d : sdf.samples) {
			double[] nd = Arrays.copyOf(d, d.length-1);
					
			samples.add(nd);
			desired.add(new double[]{d[d.length-1]});
		}
						
		final int[] fa = new int[samples.get(0).length-2]; // omit geo-vars
		for( int i = 0; i < fa.length; i++ )
			fa[i] = i+2;
		final int[] ga = new int[] { 0, 1 };
		
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);

		DataUtils.transform(samples, fa, transform.zScore);
		DataUtils.transform(desired, new int[]{0}, transform.zScore);
		
		// ------------------------------------------------------------------------

		for( final int T_MAX : new int[]{ 120000 } )	
			//for( final int nrNeurons : new int[]{ 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64 } ) 		
			//for( final int nrNeurons : new int[]{ 16,48,512,1024,2048,4096 } )
			for( final int nrNeurons : new int[]{ 64 } )
			for( final double nbInit : new double[]{ (double)nrNeurons*2.0/3.0 })
			for( final double nbFinal : new double[]{ 1.0 })	
			for( final double lr1Init : new double[]{ 0.4 }) 
			for( final double lr1Final : new double[]{ 0.005 })
			for( final double lr2Init : new double[]{ 0.2 })
			for( final double lr2Final : new double[]{ lr1Final })
			//for( int l : new int[]{1,(int)(nrNeurons/3.0),nrNeurons} )
			for( int l = 1; l <= nrNeurons/2.0; l++ )
			{	
				final int L = l;
				
				ExecutorService es = Executors.newFixedThreadPool(4);
				List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

				for (int run = 0; run < 32; run++) {
					
					int samplesSize = samples.size();
					final List<double[]> samplesTrain = new ArrayList<double[]>(samples);
					final List<double[]> desiredTrain = new ArrayList<double[]>(desired);

					final List<double[]> samplesVal = new ArrayList<double[]>();
					final List<double[]> desiredVal = new ArrayList<double[]>();
					while (samplesVal.size() < 0.3 * samplesSize) {
						int idx = r.nextInt(samplesTrain.size());
						samplesVal.add(samplesTrain.remove(idx));
						desiredVal.add(desiredTrain.remove(idx));
					}

					futures.add(es.submit(new Callable<double[]>() {

						@Override
						public double[] call() throws Exception {

							DecayFunction nbRate = new PowerDecay(nbInit, nbFinal);
							DecayFunction lrRate1 = new PowerDecay(lr1Init, lr1Final);
							DecayFunction lrRate2 = new PowerDecay(lr2Init, lr2Final);

							List<double[]> neurons = new ArrayList<double[]>();
							for (int i = 0; i < nrNeurons; i++) {
								double[] d = samplesTrain.get(r.nextInt(samplesTrain.size()));
								neurons.add(Arrays.copyOf(d, d.length));
							}
							
							Sorter<double[]> sorter = new KangasSorter<>(new DefaultSorter<>(gDist), new DefaultSorter<>(fDist), L);
							LLMNG ng = new LLMNG( neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, fa, 1 );
							
							for (int t = 0; t < T_MAX; t++) {
								int idx = r.nextInt(samplesTrain.size());
								ng.train((double) t / T_MAX, samplesTrain.get(idx));
							}
							Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samplesTrain, neurons, sorter);
																					
							// RBFN
							Map<double[], Double> hidden = new HashMap<double[], Double>();
							for (double[] c : bmus.keySet() ) {
								double d = Double.MAX_VALUE;
								for (double[] n :  bmus.keySet() )
									if (c != n)
										d = Math.min(d, fDist.dist(c, n))*1.1;
								hidden.put(c, d);
							}
							RBF rbf = new RBF(hidden, 1, fDist, 0.05);
							for (int i = 0; i < T_MAX; i++) {
								int j = r.nextInt(samplesTrain.size());
								rbf.train(samplesTrain.get(j), desiredTrain.get(j));
							}
																					
							// LM, cluster as dummy variable
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
								List<double[]> responseValLM = new ArrayList<double[]>();
								List<double[]> responseValLLM = new ArrayList<double[]>();
								List<double[]> responseValRBF = new ArrayList<double[]>();
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

									responseValLM.add(new double[] { p });
									responseValLLM.add( ng.present(d) );
									responseValRBF.add( rbf.present(d) );
								}
								return new double[]{ 
										Meuse.getRMSE( responseValLLM, desiredVal ),
										Meuse.getRMSE( responseValLM, desiredVal ),
										Meuse.getRMSE( responseValRBF, desiredVal ),
									};
							} catch (SingularMatrixException e) {
								log.debug(e.getMessage());
								System.exit(1);
							}
						return null;
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
						Files.write(Paths.get(fn), ("t_max,nrNeurons,nbInit,nbFinal,lr1Init,lr1Final,lr2Init,lr2Final,l,rmseLLM,rmseLM,rmseRBF\n").getBytes());
					}
					String s = T_MAX+","+nrNeurons+","+nbInit+","+nbFinal+","+lr1Init+","+lr1Final+","+lr2Init+","+lr2Final+","+l;
					for (int i = 0; i < ds.length; i++)
						s += ","+ds[i].getMean();
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
