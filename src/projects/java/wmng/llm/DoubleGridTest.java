package wmng.llm;

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
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.GridPos;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;

public class DoubleGridTest {
	
	private static Logger log = Logger.getLogger(DoubleGridTest.class);
	
	enum model {WMNG,CNG};
		
	public static void main(String[] args) {
		long timeAll = System.currentTimeMillis();
		
		final Random r = new Random();
								
		int maxRun = 32;
		final int[] fa = new int[]{2};
		final int[] ga = new int[] { 0, 1 };
				
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
						
		class GridData {
			Grid2D<double[]> gridTrain, gridVal;
			Map<double[], Map<double[], Double>> dMapTrain, dMapVal;
			List<double[]> samplesTrain = new ArrayList<double[]>();
			List<double[]> desiredTrain = new ArrayList<double[]>();
			List<double[]> samplesVal = new ArrayList<double[]>();
			List<double[]> desiredVal = new ArrayList<double[]>();
			
			public GridData(boolean first, boolean zScore) {				
				if( first )
					gridTrain = DoubleGrid2DUtils.create1stOrderDoubleGrid(50, 50, 3, true);
				else {
					//gridTrain = DoubleGrid2DUtils.createSpDepGrid2(50, 50, 3, true);
					gridTrain = DoubleGrid2DUtils.createSpDepGrid(50, 50, 3, true);
				}
				
				dMapTrain = GeoUtils.getRowNormedMatrix(GeoUtils.listsToWeights(GeoUtils.getNeighborsFromGrid(gridTrain)));
				for( double[] d : gridTrain.getPrototypes() ) {
					samplesTrain.add(d);
					desiredTrain.add(new double[]{d[3]});
				}
				if( zScore ) {
					DataUtils.zScoreColumns(samplesTrain, fa);
					DataUtils.zScoreColumn(desiredTrain, 0);
				}
				if( first )
					gridVal = DoubleGrid2DUtils.create1stOrderDoubleGrid(50, 50, 3, true);
				else
					gridVal = DoubleGrid2DUtils.createSpDepGrid(50, 50, 3, true);
				
				dMapVal = GeoUtils.getRowNormedMatrix(GeoUtils.listsToWeights(GeoUtils.getNeighborsFromGrid(gridVal)));
				for( double[] d : gridVal.getPrototypes() ) {
					samplesVal.add(d);
					desiredVal.add(new double[]{d[d.length-1]});
				}
				if( zScore ) {
					DataUtils.zScoreColumns(samplesVal, fa);
					DataUtils.zScoreColumn(desiredVal, 0);
				}
			}
		}
		
		for( final boolean first : new boolean[]{ false } ) {
			
			String fn = "output/resultDoubleTest_"+first+"_"+maxRun+".csv";
			try {
				Files.write(Paths.get(fn), ("model,t_max,nrNeurons,nbInit,nbFinal,lr1Init,lr1Final,lr2Init,lr2Final,alpha,beta,gamma,l,rmse\n").getBytes());
			} catch (IOException e) {
				e.printStackTrace();
			}
		
			long timeData = System.currentTimeMillis();
			final List<GridData> gridData = new ArrayList<GridData>();
			for( int i = 0; i < maxRun; i++ )
				gridData.add( new GridData( first, true ) );
			log.debug("Data took: "+(System.currentTimeMillis()-timeData)/1000.0/60.0+" min");
											
			for( final int T_MAX : new int[]{ 80000 } )	
			for( final int nrNeurons : new int[]{ 8 } ) 		
			for( final double nbInit : new double[]{ (double)nrNeurons*2.0/3.0 })
			for( final double nbFinal : new double[]{ 0.1 })	
			for( final double lr1Init : new double[]{ 0.6 }) 
			for( final double lr1Final : new double[]{ 0.01 })
			for( final double lr2Init : new double[]{ 0.4 })
			for( final double lr2Final : new double[]{ 0.01 })
			{			
				List<Object[]> models = new ArrayList<Object[]>();
				for( int l = 1; l <= nrNeurons; l++ )
					;//models.add( new Object[]{model.CNG,null,null,null,l} );
				
				for( final double alpha : new double[]{ 0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 } )
				for( final double beta :  new double[]{ 0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 } )
				for( final boolean useCtx : new boolean[]{ true, false } )
					;//models.add( new Object[]{model.WMNG, alpha, beta,useCtx,null} );
				models.add( new Object[]{model.WMNG, 0.7, 0.7, true ,null} );
								
				for( final Object[] m : models ) {
				
				ExecutorService es = Executors.newFixedThreadPool(4);
				Map<GridData,Future<List<double[]>>> futures = new HashMap<GridData,Future<List<double[]>>>();
				
				for (final GridData data : gridData ) {
					
					futures.put( data, es.submit(new Callable<List<double[]>>() {
	
						@Override
						public List<double[]> call() throws Exception {
							
							List<double[]> samplesTrain = data.samplesTrain;
							List<double[]> desiredTrain = data.desiredTrain;
													
							DecayFunction nbRate = new PowerDecay(nbInit, nbFinal);
							DecayFunction lrRate1 = new PowerDecay(lr1Init, lr1Final);
							DecayFunction lrRate2 = new PowerDecay(lr2Init, lr2Final);
							
							if( m[0] == model.WMNG ) { // WMNG + LLM
								double alpha = (double)m[1];
								double beta = (double)m[2];
								boolean useCtx = (boolean)m[3];
													
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
											
								double[] f = neurons.get(0);
								
								ContextNG_LLM ng = new ContextNG_LLM(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, fa, 1);
								for (int t = 0; t < T_MAX; t++) {
									int j = r.nextInt(samplesTrain.size());
									ng.train((double) t / T_MAX, samplesTrain.get(j), desiredTrain.get(j));
									
									//if( t % 100 == 0 )
									//	System.out.println("w/wc: "+f[2]+","+f[6]+", m/mc: "+ng.matrix.get(f)[0][2]+","+ng.matrix.get(f)[0][6] );
								}
								ng.useCtx = useCtx;
								
								//System.out.println("final, w/wc:"+f[2]+","+f[6]+", m/mc: "+ng.matrix.get(f)[0][2]+","+ng.matrix.get(f)[0][6] );
								//System.exit(1);
								
								// ------------------------------------------
								
								final List<double[]> samplesVal = data.samplesVal;
								Map<double[], Map<double[], Double>> dMapVal = data.dMapVal;
								sorter.setWeightMatrix(dMapVal); // new weight-matrix
																						
								bmuHist.clear();
								for (double[] d : samplesVal)
									bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));
										
								for( int i = 0; i < 50; i++ ) // train histMap
									for( double[] x : samplesVal )
										sorter.sort(x, neurons);
														
								List<double[]> responseVal = new ArrayList<double[]>();
								for (double[] x : samplesVal)
									responseVal.add(ng.present(x));	
																																		
								return responseVal;
							} else { // CNG+LLM
								int l = (int)m[4];
								List<double[]> neurons = new ArrayList<double[]>();
								for (int i = 0; i < nrNeurons; i++) {
									double[] d = samplesTrain.get(r.nextInt(samplesTrain.size()));
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
									int j = r.nextInt(samplesTrain.size());
									ng.train((double) t / T_MAX, samplesTrain.get(j), desiredTrain.get(j));
								}
								
								/*Map<double[],Set<double[]>> m = NGUtils.getBmuMapping(samplesTrain, neurons, sorter);
								List<GridPos> pos = new ArrayList<GridPos>();
								for( double[] d : samplesTrain )
									pos.add( data.gridTrain.getPositionOf(d) );
								Drawer.geoDrawCluster(m.values(), samplesTrain, DoubleGrid2DUtils.getGeoms(pos), "output/cng_"+l+".png", true);*/
								
								List<double[]> responseVal = new ArrayList<double[]>();
								for (double[] x : data.samplesVal)
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
					for (Entry<GridData, Future<List<double[]>>> ff : futures.entrySet() ) {
				
						List<double[]> response = ff.getValue().get();
						List<double[]> desired = ff.getKey().desiredVal;
																					
						double[] ee = new double[]{ Meuse.getRMSE(response, desired) };
						if (ds == null) {
							ds = new DescriptiveStatistics[ee.length];
							for (int i = 0; i < ee.length; i++)
								ds[i] = new DescriptiveStatistics();
						}
						for (int i = 0; i < ee.length; i++)
							ds[i].addValue(ee[i]);
					}
								
					// write statistics
					String s = (model)m[0]+","+T_MAX+","+nrNeurons+","+nbInit+","+nbFinal+","+lr1Init+","+lr1Final+","+lr2Init+","+lr2Final+","+(m[1]==null ? "" : (double)m[1])+","+(m[2]==null ? "" : (double)m[2])+","+(m[3]==null ? "" : (boolean)m[3])+","+(m[4]==null ? "" : (int)m[4]);
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
			}
			}
			log.debug("took: "+(System.currentTimeMillis()-timeAll)/1000.0/60.0+" min");
		}
	}
	
	public static void contextNGLLMtoString(ContextNG_LLM ng ) {
		List<double[]> neurons = ng.getNeurons();
		System.out.println("Prototypes: ");
		for( double[] n : neurons  )
			System.out.println("w: "+n[2]+", wc: "+n[6]);
		System.out.println("m:");
		for( double[] n : neurons  ) {
			double[] m = ng.matrix.get(n)[0];
			System.out.println("m: "+m[2]+", mc:"+m[6]);
		}
	}
}
