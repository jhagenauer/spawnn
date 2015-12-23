package wmng.llm;

import java.awt.Color;
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

import llm.LLMNG;

import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.log4j.Logger;
import org.opengis.referencing.crs.CoordinateReferenceSystem;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.sorter.SorterWMC;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.ColorUtils;
import spawnn.utils.ColorUtils.ColorMode;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.transform;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;

public class BostonTest {
	
	private static Logger log = Logger.getLogger(BostonTest.class);
	
	enum model {WMNG, NG, NG_LAG, LM};
		
	public static void main(String[] args) {
		long timeAll = System.currentTimeMillis();
		final Random r = new Random();
		
		int threads = 4;
		int maxRun = 8; 
				
		String fn = "output/resultBostonTest_"+maxRun+".csv";
		fn = fn.replaceAll(" ","");
		try {
			Files.write(Paths.get(fn), ("vars,nr_vars,t_max,nrNeurons,nbInit,nbFinal,lr1Init,lr1Final,lr2Init,lr2Final,model,alpha,beta,rmse\n").getBytes());
		} catch (IOException e) {
			e.printStackTrace();
		}
			
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/boston/boston_tracts.shp"), true);
		DataUtils.transform(sdf.samples, new int[]{5,12,13,17}, transform.log); // without transform performance is bad
		DataUtils.transform(sdf.samples, new int[]{9,10}, transform.pow2);
		DataUtils.transform(sdf.samples, transform.zScore);
				
		for( int[] vars : new int[][]{
				
				//TODO get bets nr of neurons for single var
				
				//new int[]{17}, // best single
				//new int[]{6,7,8,9,10,11,12,13,14,15,16,17,18} // all
				
				// FW select WMNG, r4, 16n
				/*new int[]{17},
				new int[]{17,10},
				new int[]{17,10,6},
				new int[]{17,10,6,16},
				new int[]{17,10,6,16,12},
				new int[]{17,10,6,16,12,11},
				new int[]{17,10,6,16,12,11,15},
				new int[]{17,10,6,16,12,11,15,9},
				new int[]{17,10,6,16,12,11,15,9,18},
				new int[]{17,10,6,16,12,11,15,9,18,14},
				new int[]{17,10,6,16,12,11,15,9,18,14,13},
				new int[]{17,10,6,16,12,11,15,9,18,14,13,8},
				new int[]{17,10,6,16,12,11,15,9,18,14,13,8,7},*/
				
				// FW select, LagNG, 64r, 16n
				new int[]{17},
				new int[]{17,10},
				new int[]{17,10,14},
				new int[]{17,10,14,16},
				new int[]{17,10,14,16,6},
				new int[]{17,10,14,16,6,15},
				new int[]{17,10,14,16,6,15,11},
				new int[]{17,10,14,16,6,15,11,12},
				new int[]{17,10,14,16,6,15,11,12,9},
				new int[]{17,10,14,16,6,15,11,12,9,18},
				new int[]{17,10,14,16,6,15,11,12,9,18,8},
				new int[]{17,10,14,16,6,15,11,12,9,18,8,13},
				new int[]{17,10,14,16,6,15,11,12,9,18,8,13,7},	
		} ) {
		long timeVar = System.currentTimeMillis();
							
		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();
		for (double[] d : sdf.samples) {
			double[] nd = new double[vars.length];
			for( int i = 0; i < vars.length; i++ )
				nd[i] = d[vars[i]];
			samples.add(nd);
			desired.add(new double[]{d[5]});
		}
		
		String vNames = "";
		for( int i = 0; i < vars.length; i++) {
			vNames += sdf.names.get(vars[i]);
			if( i < vars.length-1 )
				vNames += ";";
		}
				
		Map<double[],Map<double[],Double>> dMap = GeoUtils.getRowNormedMatrix( GeoUtils.listsToWeights(GeoUtils.getContiguityMap(samples, sdf.geoms, false, false)));
						
		final int[] fa = new int[samples.get(0).length];
		for( int i = 0; i < fa.length; i++ )
			fa[i] = i;
										
		final Dist<double[]> fDist = new EuclideanDist(fa);
				
		final TrainingData hd = new TrainingData( samples, desired, dMap );
		
		for( final int T_MAX : new int[]{ 120000 } )	
		//for( final int nrNeurons : new int[]{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36 } ) 		
		for( final int nrNeurons : new int[]{ 16 } ) 
		for( final double nbInit : new double[]{ (double)nrNeurons*2.0/3.0 })
		for( final double nbFinal : new double[]{ 1.0 })	
		for( final double lr1Init : new double[]{ 0.6 }) 
		for( final double lr1Final : new double[]{ 0.001 })
		for( final double lr2Init : new double[]{ 0.1 })
		for( final double lr2Final : new double[]{ 0.001 })
		{			
			List<Object[]> models = new ArrayList<Object[]>();
			//models.add( new Object[]{model.LM,null,null} );
			//models.add( new Object[]{model.NG,null,null} );
			models.add( new Object[]{model.NG_LAG,null,null,} );
			
			for (double alpha = 0.0; alpha <= 1; alpha = (double)Math.round( (alpha+0.05) * 100000) / 100000 )
			for (double beta = 0; beta <= 1; beta = (double)Math.round( (beta+0.05) * 100000) / 100000 )
				models.add( new Object[]{model.WMNG, alpha, beta} );

			//models.add( new Object[]{model.WMNG, 0.65, 0.25} ); // best 16n, 1v
			log.debug("models: "+models.size());
			
			Map<Object[],List<Double>> meanResiMap = new HashMap<Object[],List<Double>>();
									
			for( final Object[] m : models ) {
			long time = System.currentTimeMillis();
			
			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<List<double[]>>> futures = new ArrayList<Future<List<double[]>>>();
						
			for ( int run = 0; run < maxRun; run++ ) {
				
				futures.add( es.submit(new Callable<List<double[]>>() {

					@Override
					public List<double[]> call() throws Exception {
						
						List<double[]> samplesTrain = hd.samplesTrain;
						List<double[]> desiredTrain = hd.desiredTrain;
												
						DecayFunction nbRate = new PowerDecay(nbInit, nbFinal);
						DecayFunction lrRate1 = new PowerDecay(lr1Init, lr1Final);
						DecayFunction lrRate2 = new PowerDecay(lr2Init, lr2Final);
						
						if( m[0] == model.WMNG ) { // WMNG + LLM
							double alpha = (double)m[1];
							double beta = (double)m[2];
												
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
	
							Map<double[], Map<double[], Double>> dMapTrain = hd.dMapTrain;
							SorterWMC sorter = new SorterWMC(bmuHist, dMapTrain, fDist, alpha, beta);
								
							ContextNG_LLM ng = new ContextNG_LLM(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, fa, 1);
							ng.useCtx = true;
							for (int t = 0; t < T_MAX; t++) {
								int j = r.nextInt(samplesTrain.size());
								ng.train((double) t / T_MAX, samplesTrain.get(j), desiredTrain.get(j));
							}														
																					
							List<double[]> responseVal = new ArrayList<double[]>();
							for (double[] x : hd.samplesVal )
								responseVal.add(ng.present(x));				
																										
							return responseVal;
						} else if( m[0] == model.NG_LAG ){
							List<double[]> lagedSamplesTrain = GeoUtils.getLagedSamples(samplesTrain, hd.dMapTrain);
							
							List<double[]> neurons = new ArrayList<double[]>();
							for (int i = 0; i < nrNeurons; i++) {
								double[] d = lagedSamplesTrain.get(r.nextInt(lagedSamplesTrain.size()));
								neurons.add(Arrays.copyOf(d, d.length));
							}
							
							int[] nfa = new int[fa.length*2];
							for( int i = 0; i < fa.length; i++ ) {
								nfa[i] = fa[i];
								nfa[fa.length + i] = samplesTrain.get(0).length + fa[i];
							}
														
							Sorter<double[]> sorter = new DefaultSorter<>( new EuclideanDist(nfa));

							LLMNG ng = new LLMNG(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, nfa, 1);

							for (int t = 0; t < T_MAX; t++) {
								int j = r.nextInt(lagedSamplesTrain.size());
								ng.train((double) t / T_MAX, lagedSamplesTrain.get(j), desiredTrain.get(j));
							}

							List<double[]> lagedSamplesVal = GeoUtils.getLagedSamples(hd.samplesVal, hd.dMapVal);
							List<double[]> responseVal = new ArrayList<double[]>();
							for ( double[] x : lagedSamplesVal )
								responseVal.add(ng.present(x));
							return responseVal;
						} else if( m[0] == model.LM ){ // Linear model

							double[] y = new double[desiredTrain.size()];
							for (int i = 0; i < desiredTrain.size(); i++)
								y[i] = desiredTrain.get(i)[0];

							double[][] x = new double[samplesTrain.size()][];
							for (int i = 0; i < samplesTrain.size(); i++)
								x[i] = DoubleGridTest.getStripped(samplesTrain.get(i), fa);
							try {
								// training
								OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
								ols.setNoIntercept(false);
								ols.newSampleData(y, x);
								double[] beta = ols.estimateRegressionParameters();
								
								// testing
								List<double[]> samplesVal = hd.samplesVal;
								List<double[]> response = new ArrayList<double[]>();
								for (int i = 0; i < samplesVal.size(); i++) {
									double[] xi = DoubleGridTest.getStripped(samplesVal.get(i), fa);

									double p = beta[0]; // intercept at beta[0]
									for (int j = 1; j < beta.length; j++)
										p += beta[j] * xi[j - 1];

									response.add(new double[] { p });
								}
								return response;
							} catch (SingularMatrixException e) {
								log.debug(e.getMessage());
								System.exit(1);
							}
							return null;
						} else if( m[0] == model.NG ) {
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
							for ( double[] x : hd.samplesVal )
								responseVal.add(ng.present(x));
							return responseVal;
						} else {
							return null;
						}
					}
				}));
			}
			es.shutdown();
			
			try {
				List<Double> meanResiduals = null;
				DescriptiveStatistics ds[] = null;
				for (Future<List<double[]>> ff : futures ) {
				
						List<double[]> responseVal = ff.get();
						List<double[]> desiredVal = hd.desiredVal;
												
						double[] ee = new double[]{ Meuse.getRMSE(responseVal, desiredVal)	};
						
						if (ds == null) {
							ds = new DescriptiveStatistics[ee.length];
							for (int i = 0; i < ee.length; i++)
								ds[i] = new DescriptiveStatistics();
						}
						for (int i = 0; i < ee.length; i++)
							ds[i].addValue(ee[i]);
						
						List<Double> residuals = new ArrayList<Double>();
						for( int i = 0; i < responseVal.size(); i++ )
							residuals.add( (responseVal.get(i)[0] - desiredVal.get(i)[0])/maxRun );
						
						if( meanResiduals == null )
							meanResiduals = residuals;
						else {
							for( int i = 0; i < residuals.size(); i++ )
								meanResiduals.set( i, meanResiduals.get(i)+residuals.get(i) );
						}	
				}
				meanResiMap.put(m, meanResiduals);
				String s = vNames+","+vars.length+","+T_MAX+","+nrNeurons+","+nbInit+","+nbFinal+","+lr1Init+","+lr1Final+","+lr2Init+","+lr2Final+","+Arrays.toString(m).replaceAll("\\[", "").replaceAll("\\]", "");
				for (int i = 0; i < ds.length; i++)
					s += ","+ds[i].getMean();
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
			//log.debug("model took: "+(System.currentTimeMillis()-time)/1000.0+"s");
		}
		
		/*
		double minResi = Double.MAX_VALUE, max = Double.MIN_VALUE;
		double minResiSquared = Double.MAX_VALUE, maxResiSquared = Double.MIN_VALUE;
		for( List<Double> l : meanResiMap.values() ) {
			for( double d : l ) {
				minResi = Math.min(minResi, d);
				max = Math.max(max, d);
				
				minResiSquared = Math.min(minResiSquared, Math.pow(d,2));
				maxResiSquared = Math.max(maxResiSquared, Math.pow(d,2));
			}
		}
		
		if( models.size() < 3 )
		for( Entry<Object[], List<Double>> e : meanResiMap.entrySet() ) {
			
			List<Double> resiSquared = new ArrayList<Double>();
			for( double d : e.getValue() )
				resiSquared.add( Math.pow( d, 2));
			geoDrawValues(sdf.geoms, resiSquared, minResiSquared, maxResiSquared, sdf.crs, ColorMode.Reds, "output/residualsSquared_"+(model)e.getKey()[0]+"_"+maxRun+".png");
			
			double m = Math.max(max, Math.abs(minResi));
			geoDrawValues(sdf.geoms, e.getValue(), -m, m, sdf.crs, ColorMode.RdBu, "output/residuals_"+(model)e.getKey()[0]+"_"+maxRun+"_centered.png");
						
			List<double[]> l = new ArrayList<double[]>();
			for( double d : e.getValue() )
				l.add(new double[]{d});
			DataUtils.writeCSV( "output/residuals_"+(model)e.getKey()[0]+"_"+maxRun+".csv", l, null);
		}
						
		if( models.size() < 3 )
		for( int i = 0; i < models.size(); i++  )
			for( int j = i+1; j < models.size(); j++ ) {
				
				List<Double> resiA = meanResiMap.get(models.get(i));
				List<Double> resiB = meanResiMap.get(models.get(j));
				
				List<Double> diff = new ArrayList<Double>();
				for( int k = 0; k < resiA.size(); k++ )
					diff.add( resiA.get(k)-resiB.get(k) );
				
				double range = Math.abs(Collections.min(diff));
				range = Math.max(range, Math.abs(Collections.max(diff)));
								
				geoDrawValues(sdf.geoms, diff, -range, range, sdf.crs, ColorMode.RdBu, "output/diff_"+maxRun+"_centered.png" );
								
				List<Double> l = new ArrayList<Double>();
				for( int k = 0; k < resiA.size(); k++ ) {
					if( Math.abs(resiA.get(k)) < Math.abs(resiB.get(k)) )
						l.add( Math.abs(resiA.get(k)) );
					else
						l.add( -Math.abs(resiB.get(k)) );		
				}
				geoDrawValues(sdf.geoms, l, -range, range, sdf.crs, ColorMode.RdBu, "output/better_"+maxRun+"_centered.png" );			
			}
			*/			
		}
		log.debug("var took: "+(System.currentTimeMillis()-timeVar)/1000.0/60.0+"m");
		}
		log.debug("all took: "+(System.currentTimeMillis()-timeAll)/1000.0/60.0+"m");
		log.debug(fn+" written.");
	}
	

	public static void geoDrawValues(List<Geometry> geoms, List<Double> values, double min, double max, CoordinateReferenceSystem crs, ColorMode cm, String fn) {
		Map<Geometry, Double> doubleMap = new HashMap<Geometry, Double>();
		for (int i = 0; i < geoms.size(); i++)
			doubleMap.put(geoms.get(i), values.get(i));
		
		GeometryFactory gf = new GeometryFactory();
		Geometry minGeom = gf.createPoint(new Coordinate(0.01123153, 0.1024324));
		Geometry maxGeom = gf.createPoint(new Coordinate(0.03123412, 0.10442321));
		
		if( geoms.contains(minGeom) || geoms.contains(maxGeom))
			throw new RuntimeException("Cannot determine colors!");
		
		doubleMap.put(minGeom, min);
		doubleMap.put(maxGeom, max);
		
		Map<Geometry,Color> m = ColorUtils.getColorMap(doubleMap, cm,24,false);
		m.remove(minGeom);
		m.remove(maxGeom);
		Drawer.geoDraw( m, crs, fn);
	}
}
