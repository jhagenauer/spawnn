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

import llm.ErrorSorter;
import llm.LLMNG;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
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
import spawnn.utils.ClusterValidation;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class SimulatedRegressionOptimize {

	private static Logger log = Logger.getLogger(SimulatedRegressionOptimize.class);
	
	enum method { error, y, attr };
	
	public static void main(String[] args) {

		final Random r = new Random();
		final DecimalFormat df = new DecimalFormat("00");
		
		boolean firstWrite = true;
		
		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();
		final Map<Integer,Set<double[]>> ref = new HashMap<Integer,Set<double[]>>();
		while( samples.size() < 2000 ) {
			double lon = r.nextDouble();
			double lat = r.nextDouble();
			double x = lon;
			int c = (int)Math.round( (x-0.05)*10 );
			double coef = 2 * c/10.0;
			double y = x * coef;
			
			double[] d = new double[]{lon,lat,x,y};
			samples.add( d );
			desired.add( new double[]{y} );
			
			if( !ref.containsKey(c) )
				ref.put(c,new HashSet<double[]>());
			ref.get(c).add(d);
		}
		
		final int ta = 2;
		final int[] fa = new int[] { 2 };
		final int[] ga = new int[] { 0, 1 };
		
		// should not be necessary
		/*DataUtils.zScoreColumns(samples, fa);
		DataUtils.zScoreColumn(samples, ta);
		DataUtils.zScoreColumn(desired, 0);*/
		
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
				
		final Map<double[],Map<double[],Double>> dMap = GeoUtils.getInverseDistanceMatrix(samples, gDist, 1);
		GeoUtils.rowNormalizeMatrix(dMap);
		
		final SpatialDataFrame gwrResults = DataUtils.readSpatialDataFrameFromShapefile(new File("output/gwr.shp"), true);
		
		// ------------------------------------------------------------------------
		for( final method m : new method[]{method.error /*, method.attr*/ } )
		for( final boolean fritzkeMode : new boolean[]{ false, true } )
		for( final boolean ignSupport : new boolean[]{ false } )
		for( final int T_MAX : new int[]{ 40000 } )
		for( final double lInit : new double[]{ 2 })
		for( final int nrNeurons : new int[]{ 10 } )
		for( final double lFinal : new double[]{ 0.1, 0.01, 0.001, 0.0001 })	
		for( final double lr1Init : new double[]{ 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 })
		for( final double lr1Final : new double[]{ 0.1, 0.01, 0.001, 0.0001 })
		for( final double lr2Init : new double[]{ 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 })
		for( final double lr2Final : new double[]{ 0.1, 0.01, 0.001, 0.0001 })
		//for (int l = 1; l <= nrNeurons; l++ ) {
		for (int l : new int[]{ nrNeurons } ) {
			final int L = l;
			
			if( lr1Init <= lr1Final || lr2Init <= lr2Final )
				continue;
			
			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
	
			for (int run = 0; run < 4; run++) {
	
				futures.add(es.submit(new Callable<double[]>() {
	
					@Override
					public double[] call() throws Exception {
	
						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < nrNeurons; i++) {
							double[] d = samples.get(r.nextInt(samples.size()));
							neurons.add(Arrays.copyOf(d, d.length));
						}
						
						/*if( lr1Init == 0.0 )
							for( int i = 0; i < neurons.size(); i++ )
								neurons.get(i)[fa[0]] = 0.05+(double)i/neurons.size();*/
												
						DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
						Sorter<double[]> secSorter = null;
						if( m == method.error ) {
							secSorter = new ErrorSorter(samples, desired);
						} else if( m == method.y ) {
							secSorter = new DefaultSorter<>( new EuclideanDist(new int[]{ta} ) );
						} else if( m == method.attr ) {
							secSorter = new DefaultSorter<>( fDist );	
						} 
						Sorter<double[]> sorter = new KangasSorter<>(gSorter, secSorter, L);
	
						DecayFunction nbRate = new PowerDecay( (double)nrNeurons/lInit, lFinal);
						DecayFunction lrRate1 = new PowerDecay(lr1Init, lr1Final);
						DecayFunction lrRate2 = new PowerDecay(lr2Init, lr2Final);
						LLMNG ng = new LLMNG(neurons, 
								nbRate, lrRate1, 
								nbRate, lrRate2, 
								sorter, fa, 1 );
						ng.fritzkeMode = fritzkeMode;
						ng.ignoreSupport = ignSupport;
						
						if( m == method.error )
							((ErrorSorter)secSorter).setLLMNG(ng);
						
						for (int t = 0; t < T_MAX; t++) {
							int j = r.nextInt(samples.size());
							ng.train((double) t / T_MAX, samples.get(j), desired.get(j));
						}
						
						List<double[]> response = new ArrayList<double[]>();
						for (double[] x : samples)
							response.add(ng.present(x));
						double mse = Meuse.getMSE(response, desired);
												
						Map<double[],Double> residuals = new HashMap<double[],Double>();
						for( int i = 0; i < response.size(); i++ )
							residuals.put(samples.get(i), response.get(i)[0] - desired.get(i)[0] );
						
						//double[] moran1 = GeoUtils.getMoransIStatistics(dMap, residuals);										
						//double[] moran1 = new double[]{GeoUtils.getMoransI(dMap, residuals),0,0,0,0};
						double[] moran1 = new double[]{0,0,0,0,0};
						
						Map<double[],Set<double[]>> mapping = NGUtils.getBmuMapping(samples, neurons, sorter);
						
						Map<double[],double[]> betas = new HashMap<double[],double[]>();
						for( double[] n : mapping.keySet() ) {
							if( mapping.get(n).size() < 5 )
								continue;
							
							List<double[]> s = new ArrayList<double[]>(mapping.get(n));
							
							double[] y = new double[s.size()];
							for (int i = 0; i < s.size(); i++)
								y[i] = desired.get(samples.indexOf(s.get(i)))[0];

							double[][] x = new double[s.size()][fa.length];
							for (int i = 0; i < s.size(); i++) {
								for( int j = 0; j < fa.length; j++ )
									x[i][j] = (s.get(i)[fa[j]] /*- n[fa[j]]*/);
							}
											
							OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
							ols.setNoIntercept(false);
							ols.newSampleData(y, x);
							betas.put(n, ols.estimateRegressionParameters() );
						}
						
						//Drawer.geoDrawValues(sdf.geoms, interceptValues, null, ColorMode.Blues, "output/intercept_"+df.format(L)+".png");
						//Drawer.geoDrawValues(sdf.geoms, coefValues, null, ColorMode.Blues, "output/coef_"+df.format(L)+".png");
						
						List<Double> neuronOutputs = new ArrayList<Double>();
						List<Double> neuronCoefs = new ArrayList<Double>();
						List<Double> lnOutputs = new ArrayList<Double>();
						List<Double> lnCoefs = new ArrayList<Double>();
						for( double[] n : neurons ) {
							if( mapping.get(n).size() < 5 )
								continue;
							
							neuronOutputs.add( ng.output.get(n)[0] );
							neuronCoefs.add( ng.matrix.get(n)[0][0]);	
							
							lnOutputs.add( betas.get(n)[0] );
							lnCoefs.add( betas.get(n)[1] );	
						}
												
						//(new DefaultSorter<>( fDist )).sort(new double[]{0,0,0}, neurons);
						
						double coefError = 0;
						for( double[] n : neurons ) {
							double x = n[fa[0]];
							double coef = ng.matrix.get(n)[0][0];							
							double desiredCoef = 2 * Math.round( (x-0.05)*10 )/10.0;
							coefError += Math.pow( coef - desiredCoef, 2);
						}
						
						double pcCoefs = Double.NaN;
						if( neuronCoefs.size() > 1 )
							pcCoefs = (new PearsonsCorrelation()).correlation(toArray(neuronCoefs), toArray(lnCoefs));
						
						double pcOutputs = Double.NaN;
						if( neuronOutputs.size() > 1 )
							pcOutputs = (new PearsonsCorrelation()).correlation(toArray(neuronOutputs), toArray(lnOutputs));
																		
						return new double[] {
							Math.sqrt(mse),
							ClusterValidation.getNormalizedMutualInformation(ref.values(), mapping.values()),
							coefError,
							pcCoefs,
							pcOutputs
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
				String fn = "output/resultSynthetic.csv";
				if( firstWrite ) {
					firstWrite = false;
					Files.write(Paths.get(fn), ("method,fritzkeMode,ignSupport,t_max,nrNeurons,l,lInit,lFinal,lr1Init,lr1Final,lr2Init,lr2Final,rmse,nmi,coefError,coefCor,interCor\n").getBytes());
				}
				String s = m+","+fritzkeMode+","+ignSupport+","+T_MAX+","+nrNeurons+","+l+","+lInit+","+lFinal+","+lr1Init+","+lr1Final+","+lr2Init+","+lr2Final+"";
				for (int i = 0; i < ds.length; i++)
					s += ","+ds[i].getMean();//+","+ds[i].getStandardDeviation();
				s += "\n";
				Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
				System.out.print(s);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	public static DescriptiveStatistics getDS(List<Double> l ) {
		DescriptiveStatistics ds = new DescriptiveStatistics();
		for( double d : l )
			ds.addValue(d);
		return ds;
	}
	
	public static double[] toArray(List<Double> l ) {
		double[] r = new double[l.size()];
		for( int i = 0; i < l.size(); i++ )
			r[i] = l.get(i);
		return r;
	}
}
