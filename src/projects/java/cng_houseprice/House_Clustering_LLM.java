package cng_houseprice;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
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

import llm.LLMNG;
import llm.LLMSOM;

import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.SpatialDataFrame;

public class House_Clustering_LLM {

	private static Logger log = Logger.getLogger(House_Clustering_LLM.class);

	public static enum ClusterAlgorithm {
		geosom, wsom, som, cng, wng, ng
	};

	public static void main(String[] args) {
		int[] ga = new int[] { 0, 1 };
		final SpatialDataFrame df = DataUtils.readSpatialDataFrameFromCSV(new File("output/house_sample.csv"), ga, new int[] {}, true);
		int[] fa1 = new int[df.samples.get(0).length - 3];
		for (int i = 0; i < fa1.length; i++)
			fa1[i] = i + 2;
		int fa2 = df.samples.get(0).length - 1;

		// lnp ~ lnarea_total + lnarea_plot + age_num + cond_house_3 + heat_3 + bath_3 + attic_dum + cellar_dum + garage_3 + terr_dum + gem_kauf_index_09 + gem_abi + gem_alter_index + ln_gem_dichte + time
		// important: lnarea_tot, bez_wko, age_num
		/*int[] ga = new int[] { 0, 1 };
		int[] fa1 = new int[] { 72, 71, 75, 49, 53, 57, 58, 59, 63, 70, 17, 19, 23, 27, 74 };
		int fa2 = 1;
		final SpatialDataFrame df = DataUtils.readSpatialDataFrameFromCSV(new File("data/marco/dat4/gwr.csv"), ga, new int[] {}, true);*/
		
		final int X_DIM = 4, Y_DIM = 4; // SOM
		final int NR_NEURONS = 16; // NG
		
		log.debug("ga: " + Arrays.toString(ga));
		log.debug("fa1: " + Arrays.toString(fa1));
		log.debug("fa2: " + fa2);

		final Random r = new Random();
		final int T_MAX = 100000;

		final int threads = 4;
		final int maxK = 10; // cross-validation
		final int maxRun = 8;
		log.debug("params: "+threads + "," + maxK + "," + maxRun);

		// methods and params
		final Map<ClusterAlgorithm, List<double[]>> methods = new HashMap<ClusterAlgorithm, List<double[]>>();
		
		/*methods.put(ClusterAlgorithm.som, new ArrayList<Double>() );
		methods.get(ClusterAlgorithm.som).add( 0.0 );*/
			
		methods.put(ClusterAlgorithm.cng, new ArrayList<double[]>());
		for( int i = 1; i <= NR_NEURONS; i++ )
			methods.get(ClusterAlgorithm.cng).add( new double[]{i} );
		
		methods.put(ClusterAlgorithm.wng, new ArrayList<double[]>());		
		for( int i = 100; i >= 80; i-=2 ) {
			methods.get(ClusterAlgorithm.wng).add(
					new double[]{i}
			);
		}
		
		// [8.0, 1.0, 0.4, 0.001], RMSE: 0.35612821086827434
		/*methods.put(ClusterAlgorithm.ng, new ArrayList<double[]>());
		//(double)NR_NEURONS/3, 0.01, 0.1, 0.001,	
		for( int a : new int[]{ 8 } )
			for( double b : new double[]{ 1 } )
				for( double c : new double[]{ 1.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 } )
					for( double d : new double[]{ 0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.0 } )
						methods.get(ClusterAlgorithm.ng).add( new double[]{a,b,c,d} );
		*/
		
		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();
		for (double[] d : df.samples) {
			double[] n2 = new double[ga.length + fa1.length];
			for (int i = 0; i < ga.length; i++)
				n2[i] = d[ga[i]];
			for (int i = 0; i < fa1.length; i++)
				n2[i + ga.length] = d[fa1[i]];
			samples.add(n2);
			desired.add(new double[] { d[fa2] });
		}
				
		final int[] nfa = new int[fa1.length];
		final int[] nga = new int[ga.length];
		for (int i = 0; i < ga.length; i++)
			nga[i] = i;
		for (int i = 0; i < fa1.length; i++)
			nfa[i] = i + nga.length;

		final Dist<double[]> fDist = new EuclideanDist(nfa);
		final Dist<double[]> gDist = new EuclideanDist(nga);

		DataUtils.zScoreColumns(samples, nfa);
		DataUtils.zScoreGeoColumns(samples, ga, gDist);
		
		double bestRMSE = Double.MAX_VALUE;
		double[] bestParams = null;

		for (final ClusterAlgorithm alg : methods.keySet()) {

			for (final double[] L : methods.get(alg)) {

				ExecutorService es = Executors.newFixedThreadPool(4);
				List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

				for (int run = 0; run < maxRun; run++) {
					futures.add(es.submit(new Callable<double[]>() {

						@Override
						public double[] call() throws Exception {
							double rmse = 0;
							double r2 = 0;
							double fqe = 0;
							double sqe = 0;
							
							List<Integer> idx = new ArrayList<Integer>();
							for (int i = 0; i < samples.size(); i++)
								idx.add(i);
							Collections.shuffle(idx);

							for (int k = 0; k < maxK; k++) {

								List<double[]> training = new ArrayList<double[]>();
								List<double[]> trainingDesired = new ArrayList<double[]>();
								List<double[]> validation = new ArrayList<double[]>();
								List<double[]> validationDesired = new ArrayList<double[]>();

								for (int i = 0; i < idx.size(); i++) {
									if (k * idx.size() / maxK <= i && i < (k + 1) * idx.size() / maxK) {
										validation.add(samples.get(idx.get(i)));
										validationDesired.add(desired.get(idx.get(i)));
									} else {
										training.add(samples.get(idx.get(i)));
										trainingDesired.add(desired.get(idx.get(i)));
									}
								}
								
								if( alg == ClusterAlgorithm.geosom ) {
									Grid2D<double[]> grid = new Grid2DHex<double[]>(X_DIM, Y_DIM);
									SomUtils.initRandom(grid, samples);
	
									BmuGetter<double[]> bmuGetter = new KangasBmuGetter<double[]>(gDist, fDist, (int) L[0]);
									LLMSOM llm = new LLMSOM(new GaussKernel(new LinearDecay(grid.getMaxDist(), 0.1)), new LinearDecay(1.0, 0.005), grid, bmuGetter, new GaussKernel(new LinearDecay(grid.getMaxDist() / 2, 0.1)), new LinearDecay(0.5, 0.005), nfa, 1);
									for (int t = 0; t < T_MAX; t++) {
										int j = r.nextInt(training.size());
										llm.train((double) t / T_MAX, training.get(j), trainingDesired.get(j));
									}
	
									List<double[]> responseK = new ArrayList<double[]>();
									for (double[] x : validation)
										responseK.add(llm.present(x));
	
									rmse += Meuse.getRMSE(responseK, validationDesired);
									r2 += Math.pow(Meuse.getPearson(responseK, validationDesired), 2.0);
									
									fqe += SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples);
									sqe += SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples);
								} else if( alg == ClusterAlgorithm.wsom ) {
									Grid2D<double[]> grid = new Grid2DHex<double[]>(X_DIM, Y_DIM);
									SomUtils.initRandom(grid, samples);
	
									Map<Dist<double[]>, Double> m = new HashMap<Dist<double[]>, Double>();
									m.put(fDist, 1.0 - (double) L[0] / 100);
									m.put(gDist, (double) L[0] / 100);
									BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>(new WeightedDist<double[]>(m));
									
									LLMSOM llm = new LLMSOM(new GaussKernel(new LinearDecay(grid.getMaxDist(), 0.1)), new LinearDecay(1.0, 0.005), grid, bmuGetter, new GaussKernel(new LinearDecay(grid.getMaxDist() / 2, 0.1)), new LinearDecay(0.5, 0.005), nfa, 1);
									for (int t = 0; t < T_MAX; t++) {
										int j = r.nextInt(training.size());
										llm.train((double) t / T_MAX, training.get(j), trainingDesired.get(j));
									}
	
									List<double[]> responseK = new ArrayList<double[]>();
									for (double[] x : validation)
										responseK.add(llm.present(x));
	
									rmse += Meuse.getRMSE(responseK, validationDesired);
									r2 += Math.pow(Meuse.getPearson(responseK, validationDesired), 2.0);
									
									fqe += SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples);
									sqe += SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples);
								} else if( alg == ClusterAlgorithm.som ) {
									Grid2D<double[]> grid = new Grid2DHex<double[]>(X_DIM, Y_DIM);
									SomUtils.initRandom(grid, samples);
	
									BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>(new EuclideanDist(nfa));
									LLMSOM llm = new LLMSOM(new GaussKernel(new LinearDecay(grid.getMaxDist(), 0.1)), new LinearDecay(1.0, 0.005), grid, bmuGetter, new GaussKernel(new LinearDecay(grid.getMaxDist() / 2, 0.1)), new LinearDecay(0.5, 0.005), nfa, 1);
									for (int t = 0; t < T_MAX; t++) {
										int j = r.nextInt(training.size());
										llm.train((double) t / T_MAX, training.get(j), trainingDesired.get(j));
									}
	
									List<double[]> responseK = new ArrayList<double[]>();
									for (double[] x : validation)
										responseK.add(llm.present(x));
	
									rmse += Meuse.getRMSE(responseK, validationDesired);
									r2 += Math.pow(Meuse.getPearson(responseK, validationDesired), 2.0);
									
									fqe += SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples);
									sqe += SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples);
								} else if( alg == ClusterAlgorithm.cng ) {
									Sorter<double[]> sorter = new KangasSorter<double[]>(gDist, fDist, (int)L[0]);
									LLMNG llm = new LLMNG(NR_NEURONS, 
											//(double)NR_NEURONS/2, 0.01, 0.5, 0.005, 
											//(double)NR_NEURONS/3, 0.01, 0.1, 0.001,	
											8, 0.01, 0.5, 0.001, 
											8, 0.01, 0.4, 0.001,	
											sorter, nfa, samples.get(0).length, 1);
													
									for (int t = 0; t < T_MAX; t++) {
										int j = r.nextInt(training.size());
										llm.train((double) t / T_MAX, training.get(j), trainingDesired.get(j));
									}
									
									List<double[]> responseK = new ArrayList<double[]>();
									for (double[] x : validation)
										responseK.add(llm.present(x));
	
									rmse += Meuse.getRMSE(responseK, validationDesired);
									r2 += Math.pow(Meuse.getPearson(responseK, validationDesired), 2.0);
									
									Map<double[],Set<double[]>> mapping = NGUtils.getBmuMapping(samples, llm.getNeurons(), sorter);
									fqe += DataUtils.getMeanQuantizationError(mapping, fDist);
									sqe += DataUtils.getMeanQuantizationError(mapping, gDist);
								} else if( alg == ClusterAlgorithm.wng ) {
									
									Map<Dist<double[]>, Double> m = new HashMap<Dist<double[]>, Double>();
									m.put(fDist, 1.0 - L[0] / 100);
									m.put(gDist, L[0] / 100);
									Sorter<double[]> sorter = new DefaultSorter<double[]>(new WeightedDist<double[]>(m));
																		
									LLMNG llm = new LLMNG(NR_NEURONS, 
											8, 0.01, 0.5, 0.001, 
											8, 0.01, 0.4, 0.001,	
											/*8, 1, 0.5, 0.001, 
											8, 1, 0.5, 0.001,*/												
											sorter, nfa, samples.get(0).length, 1);
												
									for (int t = 0; t < T_MAX; t++) {
										int j = r.nextInt(training.size());
										llm.train((double) t / T_MAX, training.get(j), trainingDesired.get(j));
									}
									
									List<double[]> responseK = new ArrayList<double[]>();
									for (double[] x : validation)
										responseK.add(llm.present(x));
	
									rmse += Meuse.getRMSE(responseK, validationDesired);
									r2 += Math.pow(Meuse.getPearson(responseK, validationDesired), 2.0);
									
									Map<double[],Set<double[]>> mapping = NGUtils.getBmuMapping(samples, llm.getNeurons(), sorter);
									fqe += DataUtils.getMeanQuantizationError(mapping, fDist);
									sqe += DataUtils.getMeanQuantizationError(mapping, gDist);
								} else if( alg == ClusterAlgorithm.ng ) {
									Sorter<double[]> sorter = new DefaultSorter<double[]>(fDist);
									LLMNG llm = new LLMNG(
											//NR_NEURONS, (double)NR_NEURONS/2, 0.01, 0.5, 0.005, 
											NR_NEURONS, L[0],L[1],L[2],L[3],	
											8.0, 1.0, 0.4, 0.001,
											sorter, nfa, samples.get(0).length, 1);
																		
									for (int t = 0; t < T_MAX; t++) {
										int j = r.nextInt(training.size());
										llm.train((double) t / T_MAX, training.get(j), trainingDesired.get(j));
									}
									
									List<double[]> responseK = new ArrayList<double[]>();
									for (double[] x : validation)
										responseK.add(llm.present(x));
									
									//Map<double[],Set<double[]>> mapping = NGUtils.getBmuMapping(samples, llm.getNeurons(), sorter);
									//Drawer.geoDrawCluster(mapping.values(), samples, df.geoms, "output/cluster_"+k+".png", false);
	
									rmse += Meuse.getRMSE(responseK, validationDesired);
									r2 += Math.pow(Meuse.getPearson(responseK, validationDesired), 2.0);
									
									Map<double[],Set<double[]>> mapping = NGUtils.getBmuMapping(samples, llm.getNeurons(), sorter);
									fqe += DataUtils.getMeanQuantizationError(mapping, fDist);
									sqe += DataUtils.getMeanQuantizationError(mapping, gDist);
								}
							} 
							return new double[] { rmse / maxK, r2 / maxK, fqe/maxK, sqe/maxK};
						}
					}));
				}
				es.shutdown();
				
				double rmse = 0, r2 = 0, fqe = 0, sqe = 0;
				for( Future<double[]> f : futures ) {
					try {
						double[] d = f.get();
						rmse += d[0];
						r2 += d[1];
						fqe += d[2];
						sqe += d[3];
					} catch (InterruptedException e) {
						e.printStackTrace();
					} catch (ExecutionException e) {
						e.printStackTrace();
					}
				}
				log.debug(alg+","+Arrays.toString(L)+","+rmse/maxRun+","+r2/maxRun+","+fqe/maxRun+","+sqe/maxRun);
				
				/*if( !Double.isNaN(rmse/maxRun) && rmse/maxRun < bestRMSE ) {
					bestRMSE = rmse/maxRun;
					bestParams = L;
					log.debug("New Best: "+bestRMSE+","+Arrays.toString(L));
				}*/
			}

		}
		log.debug("Final Best: "+Arrays.toString(bestParams));
		log.debug("RMSE: "+bestRMSE);

	}
}
