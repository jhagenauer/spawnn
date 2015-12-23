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
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import llm.ErrorSorter;
import llm.LLMNG;

import org.apache.commons.math3.linear.SingularMatrixException;
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
import spawnn.utils.ColorUtils.ColorMode;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;

public class HousepriceOptimize {
	
	/* TODO Warum ist GWR besser? Wie sieht es mir RMSE/R2 aus? 
	 * TODO Können wir GWR besser annähern?
	 */

	private static Logger log = Logger.getLogger(HousepriceOptimize.class);
	
	enum method { error, coef, inter, y };

	public static void main(String[] args) {
		boolean firstWrite = true;
		final Random r = new Random();
		final DecimalFormat df = new DecimalFormat("00");
		
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
		
		DataUtils.zScoreColumns(samples, fa);
		DataUtils.zScoreColumn(desired, 0);
		/*DataUtils.zScoreGeoColumns(samples, ga, gDist); //not necessary*/
		
		final Map<double[],Map<double[],Double>> rMap = GeoUtils.getInverseDistanceMatrix(samples, gDist, 1);
		GeoUtils.rowNormalizeMatrix(rMap);
		
		
		// ------------------------------------------------------------------------

		final SpatialDataFrame gwrResults = DataUtils.readSpatialDataFrameFromShapefile(new File("output/gwr.shp"), true);
		Drawer.geoDrawValues(gwrResults, 0, ColorMode.Blues, "output/intercept_gwr.png");
		Drawer.geoDrawValues(gwrResults, 1, ColorMode.Blues, "output/coef_gwr.png");
		
		final Map<double[],Geometry> gMap = new HashMap<double[],Geometry>();
		for( int i = 0; i < gwrResults.samples.size(); i++ ) 
			gMap.put( gwrResults.samples.get(i), gwrResults.geoms.get(i) );
		
		Dist<double[]> geomDist = new Dist<double[]>() {
			@Override
			public double dist(double[] a, double[] b) {
				return gMap.get(a).distance(gMap.get(b));
			}	
		};
		
		List<double[]> bestL = null;
		double bestW = 0;
		for( List<double[]> l : GeoUtils.getKNNs(gwrResults.samples, geomDist, 6, true).values() ) {
			
			boolean toClose = false ;
			for( double[] a : l )
				for( double[] b : l )
					if( a != b && geomDist.dist(a, b) < 10 )
						toClose = true;
			if( toClose )
				continue;
			
			double wF = DataUtils.getSumOfSquares(l, new EuclideanDist( new int[]{ 0, 1} ) );
			if( bestL == null || wF < bestW ) {
				bestW = wF;
				bestL = l;
			}
		}
		
		final List<double[]> hGroup = new ArrayList<double[]>();
		for( double[] d : bestL )
			hGroup.add( samples.get( gwrResults.samples.indexOf(d)));
					
		DataUtils.writeCSV("output/hGroup.csv", hGroup, new String[]{ "x","y","area"} );
		
		// ------------------------------------------------------------------------

		try { // predict for rmse and r2
			double[] y = new double[desired.size()];
			for (int i = 0; i < desired.size(); i++)
				y[i] = desired.get(i)[0];

			double[][] x = new double[samples.size()][];
			for (int i = 0; i < samples.size(); i++)
				x[i] = stripToFA(samples.get(i), fa);
									
			// training
			OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
			ols.setNoIntercept(false);
			ols.newSampleData(y, x);
			double[] beta = ols.estimateRegressionParameters();
			
			// testing
			List<double[]> response = new ArrayList<double[]>();
			List<double[]> desiredResponse = new ArrayList<double[]>();
			Map<double[],Double> residuals = new HashMap<double[],Double>();
			for (int i = 0; i < samples.size(); i++) {
				double[] d = samples.get(i);
				double[] xi = stripToFA(d,fa);

				double p = beta[0]; // intercept at beta[0]
				for (int j = 1; j < beta.length; j++)
					p += beta[j] * xi[j - 1];

				response.add(new double[] { p });
				desiredResponse.add(desired.get(i));
				residuals.put( d, p - desired.get(i)[0] );
			}
			double[] m = GeoUtils.getMoransIStatistics(rMap, residuals); 
			log.debug("RMSE, R2, Moran, pValue: "+Meuse.getRMSE(response, desiredResponse)+","+Meuse.getR2(response, desiredResponse)+","+m[0]+","+m[4]);
		} catch (SingularMatrixException e) {
			log.debug(e.getMessage());
		}
		
		// ------------------------------------------------------------------------
		
		for( final method m : new method[]{ /*method.coef, method.inter,*/ method.error } )
		for( final int T_MAX : new int[]{ 40000 } )	
		for( final int nrNeurons : new int[]{ 8 } )
		for( final double lInit : new double[]{ nrNeurons*2.0/3 })
		for( final double lFinal : new double[]{ 0.1 })	
		for( final double lr1Init : new double[]{ 0.6 }) 
		for( final double lr1Final : new double[]{ 0.01, })
		for( final double lr2Init : new double[]{ 0.2, })
		for( final double lr2Final : new double[]{ 0.01, })
		for( final boolean ignSupport : new boolean[]{ false } )
		for( final LLMNG.mode mode : new LLMNG.mode[]{ /*LLMNG.mode.hagenauer,*/ LLMNG.mode.fritzke } )
		//for (int l : new int[]{ /*1,2,3,4,*/nrNeurons } )
		for (int l = 1; l <= nrNeurons; l++ )
		{
			final int L = l;

			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (int run = 0; run < 8; run++) {

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {

						List<Double> old = new ArrayList<Double>();
						if( m == method.coef || m == method.inter ) {
							for( int i = 0; i < samples.size(); i++ ) {
								double[] d = samples.get(i);
								old.add(d[fa[0]]);
								
								if( m == method.coef )
									d[fa[0]] = gwrResults.samples.get(i)[1];
								else if( m == method.inter )
									d[fa[0]] = gwrResults.samples.get(i)[0];
							}
						}

						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < nrNeurons; i++) {
							double[] d = samples.get(r.nextInt(samples.size()));
							neurons.add(Arrays.copyOf(d, d.length));
						}

						Sorter<double[]> secSorter = null;
						if( m == method.error )
							secSorter = new ErrorSorter(samples, desired);
						else if( m == method.coef || m == method.inter )
							secSorter = new DefaultSorter<>(fDist);
						
						DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
						Sorter<double[]> sorter = new KangasSorter<>(gSorter, secSorter, L);
						
						DecayFunction nbRate = new PowerDecay(lInit, lFinal);
						DecayFunction lrRate1 = new PowerDecay(lr1Init, lr1Final);
						DecayFunction lrRate2 = new PowerDecay(lr2Init, lr2Final);
						LLMNG ng = new LLMNG(neurons, 
								nbRate, lrRate1, 
								nbRate, lrRate2, 
								sorter, fa, 1 );
						ng.aMode = mode;
						
						if( m == method.error )
							((ErrorSorter) secSorter).setLLMNG(ng);
						
						for (int t = 0; t < T_MAX; t++) {
							int j = r.nextInt(samples.size());
							ng.train((double) t / T_MAX, samples.get(j), desired.get(j));
						}
						
						List<double[]> response = new ArrayList<double[]>();
						for (double[] x : samples)
							response.add(ng.present(x));					
						double mse =  Meuse.getMSE(response, desired);
						
						Map<double[],Set<double[]>> mapping = NGUtils.getBmuMapping(samples, neurons, sorter);
											
						Map<double[],Double> residuals = new HashMap<double[],Double>();
						for( int i = 0; i < response.size(); i++ )
							residuals.put(samples.get(i), response.get(i)[0] - desired.get(i)[0] );
						
						double rss = 0;
						for( double d : residuals.values() )
							rss += Math.pow(d, 2);
																		
						if( m == method.coef || m == method.inter ) {
							for( int i = 0; i < samples.size(); i++ ) { // restore
								double[] d = samples.get(i);
								d[fa[0]] = old.get(i);
							}
						}
									
						//double[] moran1 = GeoUtils.getMoransIStatistics(rMap, residuals);
						//double[] moran1 = new double[]{GeoUtils.getMoransI(rMap, residuals),0,0,0,0};
						double[] moran1 = new double[]{0,0,0,0,0};
						
						int nrParams = -1;
						if( L == nrNeurons )
							nrParams = nrNeurons * ( ga.length + 2 * fa.length + 1 + 1);
						else 
							nrParams = nrNeurons * ( 2 * fa.length + 1 + 1);
						
						List<Double> interceptValues = new ArrayList<Double>();
						List<Double> coefValues = new ArrayList<Double>();
						for( double[] d : samples ) 
							for( double[] n : neurons ) 
								if( mapping.get(n).contains(d)) {
									interceptValues.add( ng.output.get(n)[0] );
									coefValues.add( ng.matrix.get(n)[0][0]);
									break;
								}
						/*Drawer.geoDrawValues(sdf.geoms, interceptValues, sdf.crs, ColorMode.Blues, "output/intercept_"+mode+"_"+df.format(L)+".png");
						Drawer.geoDrawValues(sdf.geoms, coefValues, sdf.crs, ColorMode.Blues, "output/coef_"+mode+"_"+df.format(L)+".png");*/
						/*Drawer.geoDrawCluster(mapping.values(), samples, sdf.geoms, "output/cluster"+df.format(L)+".png", true);*/
						
						int sameCluster = 0;
						for( Set<double[]> s : mapping.values() )
							if( s.containsAll(hGroup) ) {
								sameCluster = 1;
								break;
							}
												
						return new double[] { 
								Math.sqrt(mse), // rmse
								
								Math.abs(moran1[0]), // moran
								moran1[4], // p-Value
								
								Meuse.getAIC(mse, nrParams, samples.size() ),
								Meuse.getAICc(mse, nrParams, samples.size() ),
								Meuse.getBIC(mse, nrParams, samples.size() ),
								
								getPearsonCor(gwrResults.samples, 0, interceptValues),
								getPearsonCor(gwrResults.samples, 1, coefValues),
								
								sameCluster,
								
								rss
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
					Files.write(Paths.get(fn), ("method,fritzke,ignSupport,t_max,nrNeurons,l,lInit,lFinal,lr1Init,lr1Final,lr2Init,lr2Final,rmse,moran,pValue,wcssF,aic,aicc,bic,corIntercept,corCoefs,sameCluster,rss\n").getBytes());
				}
				String s = m+","+mode+","+ignSupport+","+T_MAX+","+nrNeurons+","+l+","+lInit+","+lFinal+","+lr1Init+","+lr1Final+","+lr2Init+","+lr2Final;
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
	
	public static double getPearsonCor(List<double[]> a, int aIdx, List<Double> b ) {
		double[] aa = new double[a.size()];
		for( int i = 0; i < a.size(); i++ )
			aa[i] = a.get(i)[aIdx];
		
		double[] bb = new double[b.size()];
		for( int i = 0; i < b.size(); i++ )
			bb[i] = b.get(i);
		return (new PearsonsCorrelation()).correlation(aa, bb);
	}
	
	public static DescriptiveStatistics getDS(List<Double> l ) {
		DescriptiveStatistics ds = new DescriptiveStatistics();
		for( double d : l )
			ds.addValue(d);
		return ds;
	}
}
