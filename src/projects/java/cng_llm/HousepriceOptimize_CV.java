package cng_llm;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.text.DecimalFormat;
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

import llm.ErrorSorter;
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
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class HousepriceOptimize_CV {

	private static Logger log = Logger.getLogger(HousepriceOptimize_CV.class);

	public static void main(String[] args) {
		boolean firstWrite = true;
		final Random r = new Random();
		final DecimalFormat df = new DecimalFormat("00");
		
		final List<double[]> samplesOrig = new ArrayList<double[]>();
		final List<double[]> desiredOrig = new ArrayList<double[]>();

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("data/marco/dat4/gwr.csv"), new int[] { 6, 7 }, new int[] {}, true);
		List<String> vars = new ArrayList<String>();
		vars.add("xco");
		vars.add("yco");

		vars.add("lnarea_tot");
		vars.add("lnarea_plo");
		vars.add("attic_dum");
		vars.add("cellar_dum");
		vars.add("cond_house_3");
		vars.add("heat_3");
		vars.add("bath_3");
		vars.add("garage_3");
		vars.add("terr_dum");
		vars.add("age_num");
		vars.add("time_index");
		//vars.add("zsp_alq_09");
		//vars.add("gem_kauf_i");
		//vars.add("gem_abi");
		//vars.add("gem_alter_");
		//vars.add("ln_gem_dic");

		final int da = sdf.names.indexOf("lnp");
		for (double[] d : sdf.samples) {
			if (d[sdf.names.indexOf("time_index")] < 6)
				continue;
			double[] nd = new double[vars.size()];
			
			for (int i = 0; i < nd.length; i++)
				nd[i] = d[sdf.names.indexOf(vars.get(i))]; 
			
			// jitter
			nd[0] += 0.02+r.nextDouble()*0.01;
			nd[1] += 0.02+r.nextDouble()*0.01;
					
			samplesOrig.add(nd);
			desiredOrig.add(new double[]{d[da]});
		}

		final int[] fa = new int[vars.size()-2]; // omit geo-vars
		for( int i = 0; i < fa.length; i++ )
			fa[i] = i+2;
		final int[] ga = new int[] { 0, 1 };
		
		DataUtils.zScoreColumns(samplesOrig, fa);
		DataUtils.zScoreColumn(desiredOrig, 0); // should not be necessary
				
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		// ------------------------------------------------------------------------
				
		// Ohne ignore ist es etwas besser
		for( final int T_MAX : new int[]{ 40000 } )
		for( final int nrNeurons : new int[]{ 16 } )
		for( final double lInit : new double[]{ nrNeurons/2 })
		for( final double lFinal : new double[]{ 0.1 })	
		for( final double lr1Init : new double[]{ 0.5 })
		for( final double lr1Final : new double[]{ 0.001 })
		for( final double lr2Init : new double[]{ 0.1 })
		for( final double lr2Final : new double[]{ 0.001 })
		for( final LLMNG.mode mode : new LLMNG.mode[]{ LLMNG.mode.fritzke } )
		for (int l = 1; l <= nrNeurons; l++ ) {
			final int L = l;

			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (int run = 0; run < 64; run++) {

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
						
						List<double[]> samples = new ArrayList<double[]>(samplesOrig);
						List<double[]> desired = new ArrayList<double[]>(desiredOrig);

						List<double[]> samplesVal = new ArrayList<double[]>();
						List<double[]> desiredVal = new ArrayList<double[]>();
						while( samplesVal.size() < samples.size()/3 ) {
							int idx = r.nextInt(samples.size());
							samplesVal.add( samples.remove(idx));
							desiredVal.add( desired.remove(idx));
						}

						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < nrNeurons; i++) {
							double[] d = samples.get(r.nextInt(samples.size()));
							neurons.add(Arrays.copyOf(d, d.length));
						}

						ErrorSorter errorSorter = new ErrorSorter(samples, desired);
						DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
						Sorter<double[]> sorter = new KangasSorter<>(gSorter, errorSorter, L);

						DecayFunction nbRate = new PowerDecay(/*neurons.size()/*/lInit, lFinal);
						DecayFunction lrRate1 = new PowerDecay(lr1Init, lr1Final);
						DecayFunction lrRate2 = new PowerDecay(lr2Init, lr2Final);
						LLMNG ng = new LLMNG(neurons, 
								nbRate, lrRate1, 
								nbRate, lrRate2, 
								sorter, fa, 1 );
						errorSorter.setLLMNG(ng);
						ng.aMode = mode;

						for (int t = 0; t < T_MAX; t++) {
							int j = r.nextInt(samples.size());
							ng.train((double) t / T_MAX, samples.get(j), desired.get(j));
						}
						
						List<double[]> response = new ArrayList<double[]>();
						for (double[] x : samples)
							response.add(ng.present(x));					
						double rmse =  Meuse.getRMSE(response, desired);
												
						Map<double[],Double> residuals = new HashMap<double[],Double>();
						for( double[] d : samples ) {
							int idx = samples.indexOf(d);	
							residuals.put(d, ng.present(d)[0] - desired.get(idx)[0] );
						}
						
						DefaultSorter<double[]> fSorter = new DefaultSorter<>(fDist);
						Sorter<double[]> sorter2 = new KangasSorter<>(gSorter, fSorter, L); 
						ng.setSorter( sorter2 );
																			
						List<double[]> responseVal = new ArrayList<double[]>();
						for( double[] x : samplesVal )
							responseVal.add( ng.present(x) );
						double rmse2 =  Meuse.getRMSE(responseVal, desiredVal);
							
						List<double[]> sortedNeurons = new ArrayList<double[]>(neurons);
						double rmse3 = -1;
						try { // predict for rmse and r2
							double[] y = new double[desired.size()];
							for (int i = 0; i < desired.size(); i++)
								y[i] = desired.get(i)[0];

							double[][] x = new double[samples.size()][];
							for (int i = 0; i < samples.size(); i++) {
								double[] d = samples.get(i);
								x[i] = getStripped(d, fa);
								x[i] = Arrays.copyOf(x[i], x[i].length+1);
								sorter2.sort(d, neurons);
								x[i][x[i].length-1] = sortedNeurons.indexOf( neurons.get(0) ); 
							}
													
							// training
							OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
							ols.setNoIntercept(false);
							ols.newSampleData(y, x);
							double[] beta = ols.estimateRegressionParameters();
							
							// testing
							List<double[]> responseVal2 = new ArrayList<double[]>();
							for (int i = 0; i < samplesVal.size(); i++) {
								double[] d = samplesVal.get(i);															
								double[] xi = getStripped(d,fa);
								xi = Arrays.copyOf(xi, xi.length+1);
								sorter2.sort(d, neurons);
								xi[xi.length-1] = sortedNeurons.indexOf( neurons.get(0) ); 
								
								double p = beta[0]; // intercept at beta[0]
								for (int j = 1; j < beta.length; j++)
									p += beta[j] * xi[j - 1];

								responseVal2.add(new double[] { p });
							}
							rmse3 = Meuse.getRMSE(responseVal2, desiredVal);
							
						} catch (SingularMatrixException e) {
							log.debug(e.getMessage());
						}
												
						return new double[] { 
								rmse,
								rmse2,
								rmse3
								};
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
				if( firstWrite ) {
					firstWrite = false;
					Files.write(Paths.get(fn), ("t_max,nrNeurons,l,lInit,lFinal,lr1Init,lr1Final,lr2Init,lr2Final,rmse,rmse_sd,rmse2,rmse2_sd,rmse3,rmse3_sd\n").getBytes());
				}
				String s = T_MAX+","+nrNeurons+","+l+","+lInit+","+lFinal+","+lr1Init+","+lr1Final+","+lr2Init+","+lr2Final+"";
				for (int i = 0; i < ds.length; i++)
					s += ","+ds[i].getMean()+","+ds[i].getStandardDeviation();
				s += "\n";
				Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
				System.out.print(s);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	public static double[] getStripped( double[] d, int[] fa ) {
		double[] nd = new double[fa.length];
		for( int i = 0; i < fa.length; i++ )
			nd[i] = d[fa[i]];
		return nd;
	}
}
