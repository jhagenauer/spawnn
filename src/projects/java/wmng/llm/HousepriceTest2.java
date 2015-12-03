package wmng.llm;

import java.io.File;
import java.text.DecimalFormat;
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

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.sorter.SorterWMC;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.ColorBrewerUtil.ColorMode;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;

public class HousepriceTest2 {
	
	// TODO it would be great of we can determine significant error-differences via monte-carlo
	
	private static Logger log = Logger.getLogger(HousepriceTest2.class);
	
	enum model {WMNG,CNG};
		
	public static void main(String[] args) {
		long timeAll = System.currentTimeMillis();
		final Random r = new Random();
		final DecimalFormat df = new DecimalFormat("00");
								
		int maxRun = 1;
				
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("output/houseprice_all.csv"), new int[] { 0,1 }, new int[] {}, true);
		final List<String> names = sdf.names;
		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();
		final List<Geometry> geoms = sdf.geoms;
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
		//DataUtils.zScoreGeoColumns(samples, ga, gDist); // should have no affect 
				
		long timeData = System.currentTimeMillis();
		final Map<double[],Map<double[],Double>> dMap = GeoUtils.getRowNormedMatrix( GeoUtils.listsToWeights( GeoUtils.getKNNs( samples, gDist, 8, false ) ) );
		//final Map<double[],Map<double[],Double>> dMap =  GeoUtils.getRowNormedMatrix(GeoUtils.getInverseDistanceMatrix(samples, gDist, 2, 20000) );		
		log.debug("Dist. mat. took: "+(System.currentTimeMillis()-timeData)/1000.0/60.0+" min");
		
		// Interessant sind:
		// wmng besser: 1353, 2980, 1757
		// cng besser: 353, 2128
		List<double[]> interesting = new ArrayList<double[]>();
		for( int i : new int[]{ 1353, 2980, 1757, 353, 2128 } )
			interesting.add( samples.get(i));
		
		/*int reps = 9999;
		Map<Integer,List<double[]>> lisas = new HashMap<Integer,List<double[]>>();
		for( int i : fa )
			lisas.put(i, GeoUtils.getLocalMoransIMonteCarlo(samples, i, dMap, reps) );
		Map<double[],Double> values = new HashMap<double[],Double>();
		for( double[] d : samples )
			values.put(d, desired.get(samples.indexOf(d))[0]);
		List<double[]> lisaPrice = GeoUtils.getLocalMoransIMonteCarlo(samples, values, dMap, reps);
		*/
		for( double[] d : interesting ) {
			int idx = samples.indexOf(d);
			double price = desired.get(idx)[0];
			
			double sumFDist = 0;
			double sumGeoDist = 0;
			double sumPriceDist = 0;
			int nrNbs = dMap.get(d).size();
			for( double[] nb : dMap.get(d).keySet() ) {
				sumFDist += fDist.dist(d, nb);
				sumGeoDist += gDist.dist(d, nb);
				sumPriceDist += Math.abs( price - desired.get(samples.indexOf(nb))[0]);
			}
			log.debug(samples.indexOf(d)+","+sumFDist/nrNbs+","+sumGeoDist/nrNbs+","+sumPriceDist/nrNbs);
			
			/*for( int i : fa ) {
				double[] l = lisas.get(i).get(idx);
				log.debug(sdf.names.get(i)+","+l[0]+","+l[4]);
			}
			log.debug("lnp,"+lisaPrice.get(idx)[0]+","+lisaPrice.get(idx)[4]);
			log.debug("------------------------------------------------------");*/
		}
													
		for( final int T_MAX : new int[]{ 80000 } )	
		for( final int nrNeurons : new int[]{ 16 } ) 		
		for( final double nbInit : new double[]{ (double)nrNeurons*2.0/3.0 })
		for( final double nbFinal : new double[]{ 1.0 })	
		for( final double lr1Init : new double[]{ 0.6 }) 
		for( final double lr1Final : new double[]{ 0.001 })
		for( final double lr2Init : new double[]{ 0.1 })
		for( final double lr2Final : new double[]{ 0.001 })
		{			
			List<Object[]> models = new ArrayList<Object[]>();
			for( int l : new int[]{1} )
			//for( int l = 1; l < nrNeurons; l++ )
				models.add( new Object[]{model.CNG,null,null,null,l} );
			
			/*for( final double alpha : new double[]{ 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 } )
			for( final double beta :  new double[]{ 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 } )
			for( final double gamma : new double[]{ 0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 } )
				models.add( new Object[]{model.WMNG, alpha, beta,gamma,null} );*/
			models.add( new Object[]{model.WMNG, 0.3, 0.0, 0.1,null} );
						
			Map<Object[],List<Double>> meanErrors = new HashMap<Object[],List<Double>>();
			for( final Object[] m : models ) {
			
			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<List<double[]>>> futures = new ArrayList<Future<List<double[]>>>();
			
			for ( int run = 0; run < maxRun; run++ ) {
				final int RUN = run;
				
				futures.add( es.submit(new Callable<List<double[]>>() {

					@Override
					public List<double[]> call() throws Exception {
						
						DecayFunction nbRate = new PowerDecay(nbInit, nbFinal);
						DecayFunction lrRate1 = new PowerDecay(lr1Init, lr1Final);
						DecayFunction lrRate2 = new PowerDecay(lr2Init, lr2Final);
						
						if( m[0] == model.WMNG ) { // WMNG + LLM
							double alpha = (double)m[1];
							double beta = (double)m[2];
							double gamma = (double)m[3];
												
							List<double[]> neurons = new ArrayList<double[]>();
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
	
							SorterWMC sorter = new SorterWMC(bmuHist, dMap, fDist, alpha, beta);
								
							ContextNG_LLM ng = new ContextNG_LLM(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, fa, 1);
							for (int t = 0; t < T_MAX; t++) {
								int j = r.nextInt(samples.size());
								ng.train((double) t / T_MAX, samples.get(j), desired.get(j));
							}							
							//ng.gamma = gamma;
							sorter.setHistMutable(false);
							
							// ------------------------------------------
																											
							List<double[]> response = new ArrayList<double[]>();
							for (double[] x : samples)
								response.add(ng.present(x));	
							
							if( RUN == 0 ) {
								Map<double[],Set<double[]>> m = NGUtils.getBmuMapping(samples, neurons, sorter);
								
								//Drawer.geoDrawCluster(m.values(), samples, geoms, "output/cng_"+df.format(l)+"_"+df.format(RUN)+".png", false);
								Map<double[],Double> clustRMSE = new HashMap<double[],Double>();
								for( double[] n : neurons ) {
									List<double[]> s = new ArrayList<double[]>(m.get(n));
									List<double[]> r = new ArrayList<double[]>();
									List<double[]> d = new ArrayList<double[]>();
									for( double[] x : s ) {
										d.add( desired.get(samples.indexOf(x)));
										r.add( ng.present(x) );
									}
									clustRMSE.put(n,Meuse.getRMSE(r, d));
									log.debug("Cluster "+neurons.indexOf(n)+","+clustRMSE.get(n));
								}
								
								List<double[]> ns = new ArrayList<double[]>();
								for( double[] d : samples )
									for( Entry<double[], Set<double[]>> e : m.entrySet() )
										if( e.getValue().contains(d) ) {
											double[] nd = Arrays.copyOf(d, d.length+4);
											nd[nd.length-4] = desired.get(samples.indexOf(d))[0];
											nd[nd.length-3] = ng.present(d)[0]; // prediction
											nd[nd.length-2] = neurons.indexOf(e.getKey());
											nd[nd.length-1] = clustRMSE.get(e.getKey());
											
											ns.add( nd );
										}
								List<String> nn = new ArrayList<String>(names);
								nn.add("prediction");
								nn.add("cluster");
								nn.add("rmse");
								
								DataUtils.writeShape(ns, geoms, nn.toArray(new String[0]), null, "output/wmng_"+df.format(RUN)+".shp");
							}
																																	
							return response;
						} else { // CNG+LLM
							int l = (int)m[4];
							List<double[]> neurons = new ArrayList<double[]>();
							for (int i = 0; i < nrNeurons; i++) {
								double[] d = samples.get(r.nextInt(samples.size()));
								neurons.add(Arrays.copyOf(d, d.length));
							}

							Sorter<double[]> secSorter = new DefaultSorter<>(fDist);
							DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
							Sorter<double[]> sorter = new KangasSorter<>(gSorter, secSorter, l );
							
							LLMNG ng = new LLMNG(neurons, 
									nbRate, lrRate1, 
									nbRate, lrRate2, 
									sorter, fa, 1 );
							
							for (int t = 0; t < T_MAX; t++) {
								int j = r.nextInt(samples.size());
								ng.train((double) t / T_MAX, samples.get(j), desired.get(j));
							}
							
							List<double[]> response = new ArrayList<double[]>();
							for (double[] x : samples)
								response.add(ng.present(x));
							
							if( RUN == 0 ) {
								Map<double[],Set<double[]>> m = NGUtils.getBmuMapping(samples, neurons, sorter);
								
								//Drawer.geoDrawCluster(m.values(), samples, geoms, "output/cng_"+df.format(l)+"_"+df.format(RUN)+".png", false);
								Map<double[],Double> clustRMSE = new HashMap<double[],Double>();
								for( double[] n : neurons ) {
									List<double[]> s = new ArrayList<double[]>(m.get(n));
									List<double[]> r = new ArrayList<double[]>();
									List<double[]> d = new ArrayList<double[]>();
									for( double[] x : s ) {
										d.add( desired.get(samples.indexOf(x)));
										r.add( ng.present(x) );
									}
									clustRMSE.put(n,Meuse.getRMSE(r, d));
									log.debug("Cluster "+neurons.indexOf(n)+","+clustRMSE.get(n));
								}
								
								List<double[]> ns = new ArrayList<double[]>();
								for( double[] d : samples )
									for( Entry<double[], Set<double[]>> e : m.entrySet() )
										if( e.getValue().contains(d) ) {
											double[] nd = Arrays.copyOf(d, d.length+4);
											nd[nd.length-4] = desired.get(samples.indexOf(d))[0];
											nd[nd.length-3] = ng.present(d)[0]; // prediction
											nd[nd.length-2] = neurons.indexOf(e.getKey());
											nd[nd.length-1] = clustRMSE.get(e.getKey());
											
											ns.add( nd );
										}
								List<String> nn = new ArrayList<String>(names);
								nn.add("prediction");
								nn.add("cluster");
								nn.add("rmse");
								
								DataUtils.writeShape(ns, geoms, nn.toArray(new String[0]), null, "output/cng_"+df.format(RUN)+".shp");
							}
							
							return response;
						}
					}
				}));
			}
			es.shutdown();
			
			try {
				String ms = m[0]+"_"+(m[1]==null ? "" : (double)m[1])+"_"+(m[2]==null ? "" : (double)m[2])+","+(m[3]==null ? "" : (double)m[3])+"_"+(m[4]==null ? "" : (int)m[4]);
				log.debug(ms);
				
				List<DescriptiveStatistics> errorStats = null;
				for (Future<List<double[]>> ff : futures ) {
					List<double[]> response = ff.get();
					
					List<Double> error = new ArrayList<Double>();
					for( int i = 0; i < response.size(); i++ )
						error.add( Math.pow( response.get(i)[0]-desired.get(i)[0], 2));
					
					if( errorStats == null ) {
						errorStats = new ArrayList<DescriptiveStatistics>();
						for( int i = 0; i < error.size(); i++ ) {
							DescriptiveStatistics ds = new DescriptiveStatistics();
							ds.addValue(error.get(i));
							errorStats.add(ds);
						}
					} else 
						for( int i = 0; i < error.size(); i++ )
							errorStats.get(i).addValue(error.get(i));
										
					/*double[] morans = GeoUtils.getMoransIStatistics(dMap, samples, error) ;
					log.debug("Morans: "+morans[0]+", "+morans[4] );*/
					//Drawer.geoDrawValues(geoms, error, sdf.crs, ColorMode.Reds, "output/"+ms+"_"+futures.indexOf(ff)+"_error.png");
				}
				
				log.debug(nbFinal+","+lr1Init+","+lr1Final+","+lr2Init+","+lr2Final);
				
				double sumErrorVar = 0;
				double sumError = 0;
				for( DescriptiveStatistics ds : errorStats ) {
					sumErrorVar += ds.getVariance();
					sumError += ds.getSum();
				}
				log.debug("MeanErrorSum: "+(sumError/maxRun)+", MeanErrorVar: "+(sumErrorVar/maxRun));
				
				List<Double> meanError = new ArrayList<Double>();
				for( DescriptiveStatistics ds : errorStats )
					meanError.add( ds.getMean() );
				//Drawer.geoDrawValues(geoms, meanError, sdf.crs, ColorMode.Reds, "output/"+ms+"_mean.png");
				meanErrors.put(m,meanError);
				
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
		
		for( int i = 0; i < models.size(); i++ )
			for( int j = i+1; j < models.size(); j++ ) {
				
				List<Double> m1Me = meanErrors.get(models.get(i));
				List<Double> m2Me = meanErrors.get(models.get(j));
								
				List<double[]> diff = new ArrayList<double[]>();
				for( int k = 0; k < m1Me.size(); k++ )
					diff.add( new double[]{ m1Me.get(k), m2Me.get(k), m1Me.get(k) - m2Me.get(k) } );
				
				Drawer.geoDrawValues(geoms, diff, 2, sdf.crs, ColorMode.Reds, "output/meanErrorDiff_"+i+"_"+j+".png");
				DataUtils.writeShape( diff, geoms, new String[]{"m1","m2","errorDif"}, sdf.crs, "output/meanErrorDiff_"+i+"_"+j+".shp");
			}
		}

		log.debug("took: "+(System.currentTimeMillis()-timeAll)/1000.0/60.0+" min");
	}
}
