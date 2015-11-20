package cng_houseprice;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Map.Entry;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import llm.LLMNG;

import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;
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
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class HousepriceOptimize {
	
	private static Logger log = Logger.getLogger(HousepriceOptimize.class);
	
	enum method { attr, error, coef, inter, y };

	public static void main(String[] args) {
		boolean firstWrite = true;
		final Random r = new Random();
		
		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();
		
		// single+ctx... gut, ausser für viele neuronen, dann wieder l==1
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
		
		DataUtils.zScoreColumns(samples, fa);
		DataUtils.zScoreColumn(desired, 0);
		/*DataUtils.zScoreGeoColumns(samples, ga, gDist); //not necessary*/

		for( final int T_MAX : new int[]{ 40000 } )	
		for( final int nrNeurons : new int[]{ 8 } )
		for( final double nbInit : new double[]{ (double)nrNeurons*2.0/3.0 })
		for( final double nbFinal : new double[]{ 0.1 })	
		for( final double lr1Init : new double[]{ 0.6 }) 
		for( final double lr1Final : new double[]{ 0.01 })
		for( final double lr2Init : new double[]{ 0.6, 0.1 })
		for( final double lr2Final : new double[]{ 0.01 })
		for( final boolean ignSupport : new boolean[]{ false } )
		for( final LLMNG.mode mode : new LLMNG.mode[]{ /*LLMNG.mode.hagenauer,*/ LLMNG.mode.fritzke } )
		//for (int l : new int[]{ 1,nrNeurons } )
		for (int l = 1; l <= nrNeurons; l++ )
		{
			final int L = l;

			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (int run = 0; run < 128; run++) {

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
						
						List<double[]> samplesTrain = new ArrayList<double[]>(samples);
						List<double[]> desiredTrain = new ArrayList<double[]>(desired);
						List<double[]> samplesVal = new ArrayList<double[]>();
						List<double[]> desiredVal = new ArrayList<double[]>();
						
						while( samplesVal.size() < samples.size()/3 ) {
							int idx = r.nextInt(samplesTrain.size());
							samplesVal.add(samplesTrain.remove(idx));
							desiredVal.add(desiredTrain.remove(idx));
						}
						
						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < nrNeurons; i++) {
							double[] d = samplesTrain.get(r.nextInt(samplesTrain.size()));
							neurons.add(Arrays.copyOf(d, d.length));
						}

						Sorter<double[]> secSorter = new DefaultSorter<>(fDist);
						DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
						Sorter<double[]> sorter = new KangasSorter<>(gSorter, secSorter, L);
						
						DecayFunction nbRate = new PowerDecay(nbInit, nbFinal);
						DecayFunction lrRate1 = new PowerDecay(lr1Init, lr1Final);
						DecayFunction lrRate2 = new PowerDecay(lr2Init, lr2Final);
						LLMNG ng = new LLMNG(neurons, 
								nbRate, lrRate1, 
								nbRate, lrRate2, 
								sorter, fa, 1 );
						ng.aMode = mode;
						ng.ignSupport = ignSupport;
						
						for (int t = 0; t < T_MAX; t++) {
							int j = r.nextInt(samplesTrain.size());
							ng.train((double) t / T_MAX, samplesTrain.get(j), desiredTrain.get(j));
						}
						
						List<double[]> responseValLLM = new ArrayList<double[]>();
						for (double[] x : samplesVal)
							responseValLLM.add(ng.present(x));				
						
						List<double[]> responseValLinReg = new ArrayList<double[]>();
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

									responseValLinReg.add(new double[] { p });
								}
							} catch (SingularMatrixException e) {
								log.debug(e.getMessage());
								System.exit(1);
							}
						}
													
						return new double[] { 
								Meuse.getRMSE(responseValLLM, desiredVal),
								Meuse.getRMSE(responseValLinReg, desiredVal)
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
				String fn = "output/resultHouseprice.csv";
				if( firstWrite ) {
					firstWrite = false;
					Files.write(Paths.get(fn), ("fritzke,ignSupport,t_max,nrNeurons,l,lInit,lFinal,lr1Init,lr1Final,lr2Init,lr2Final,rmseLLM,rmseLinReg\n").getBytes());
				}
				String s = mode+","+ignSupport+","+T_MAX+","+nrNeurons+","+l+","+nbInit+","+nbFinal+","+lr1Init+","+lr1Final+","+lr2Init+","+lr2Final;
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
	
	public static double[] stripToFA( double[] d, int[] fa ) {
		double[] nd = new double[fa.length];
		for( int i = 0; i < fa.length; i++ )
			nd[i] = d[fa[i]];
		return nd;
	}
	
	public static List<double[]> toDoubleArray(List<Double> l ) {
		List<double[]> nl = new ArrayList<double[]>();
		for( double d : l )
			nl.add(new double[]{d});
		return nl;
	}
	
	public static double getCorrelation(List<double[]> a, int aIdx, List<Double> b, boolean pearson ) {
		double[] aa = new double[a.size()];
		for( int i = 0; i < a.size(); i++ )
			aa[i] = a.get(i)[aIdx];
		
		double[] bb = new double[b.size()];
		for( int i = 0; i < b.size(); i++ )
			bb[i] = b.get(i);
		if( pearson )
			return (new PearsonsCorrelation()).correlation(aa, bb);
		else
			return (new SpearmansCorrelation()).correlation(aa, bb);
	}
	
	public static DescriptiveStatistics getDS(List<Double> l ) {
		DescriptiveStatistics ds = new DescriptiveStatistics();
		for( double d : l )
			ds.addValue(d);
		return ds;
	}
	
	public static double[] getStripped(double[] d, int[] fa) {
		double[] nd = new double[fa.length];
		for (int i = 0; i < fa.length; i++)
			nd[i] = d[fa[i]];
		return nd;
	}
}
