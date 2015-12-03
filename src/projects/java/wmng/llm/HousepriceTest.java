package wmng.llm;

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
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class HousepriceTest {
	
	private static Logger log = Logger.getLogger(HousepriceTest.class);
	
	enum model {WMNG,CNG};
		
	public static void main(String[] args) {
		long timeAll = System.currentTimeMillis();
		final Random r = new Random();
			
		int maxRun = 16;
			
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("output/houseprice_no_ctx.csv"), new int[] { 0,1 }, new int[] {}, true);
		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();
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
		
			String fn = "output/resultHousepriceTest_"+sdf.names+"_"+maxRun+".csv";
			fn = fn.replaceAll(" ","");
			try {
				Files.write(Paths.get(fn), ("dmMode,model,t_max,nrNeurons,nbInit,nbFinal,lr1Init,lr1Final,lr2Init,lr2Final,alpha,beta,gamma,l,rmse,rmseVar\n").getBytes());
			} catch (IOException e) {
				e.printStackTrace();
			}
				
		class HousepriceData {
			List<double[]> samplesTrain;
			List<double[]> desiredTrain;
			Map<double[],Map<double[],Double>> dMapTrain;
			List<double[]> samplesVal;
			List<double[]> desiredVal;
			Map<double[],Map<double[],Double>> dMapVal;
			
			public HousepriceData(List<double[]> samples, List<double[]>  desired ) {
				samplesTrain = new ArrayList<double[]>(samples);
				desiredTrain = new ArrayList<double[]>(desired);
				
				samplesVal = new ArrayList<double[]>();
				desiredVal = new ArrayList<double[]>();
				
				while( samplesVal.size() < samples.size()/3 ) {
					int idx = r.nextInt(samplesTrain.size());
					samplesVal.add(samplesTrain.remove(idx));
					desiredVal.add(desiredTrain.remove(idx));
				}			
			}
			
			public void setDistanceMatrices( int mode ) {
				switch(mode) {
					case 0:
						dMapVal = GeoUtils.getRowNormedMatrix( GeoUtils.listsToWeights( GeoUtils.getKNNs( samplesVal, gDist, 4, false ) ) );
						dMapTrain = GeoUtils.getRowNormedMatrix( GeoUtils.listsToWeights( GeoUtils.getKNNs( samplesTrain, gDist, 4, false ) ) );
						break;
					case 1:
						dMapVal = GeoUtils.getRowNormedMatrix( GeoUtils.listsToWeights( GeoUtils.getKNNs( samplesVal, gDist, 6, false ) ) );
						dMapTrain = GeoUtils.getRowNormedMatrix( GeoUtils.listsToWeights( GeoUtils.getKNNs( samplesTrain, gDist, 6, false ) ) );
						break;
					case 2:
						dMapVal = GeoUtils.getRowNormedMatrix( GeoUtils.listsToWeights( GeoUtils.getKNNs( samplesVal, gDist, 8, false ) ) );
						dMapTrain = GeoUtils.getRowNormedMatrix( GeoUtils.listsToWeights( GeoUtils.getKNNs( samplesTrain, gDist, 8, false ) ) );
						break;
					case 3:
						dMapVal = GeoUtils.getRowNormedMatrix( GeoUtils.listsToWeights( GeoUtils.getKNNs( samplesVal, gDist, 12, false ) ) );
						dMapTrain = GeoUtils.getRowNormedMatrix( GeoUtils.listsToWeights( GeoUtils.getKNNs( samplesTrain, gDist, 12, false ) ) );
						break;
					case 4:
						dMapVal = GeoUtils.getRowNormedMatrix(GeoUtils.getInverseDistanceMatrix(samplesVal, gDist, 1, 20000) );
						dMapTrain = GeoUtils.getRowNormedMatrix(GeoUtils.getInverseDistanceMatrix(samplesTrain, gDist, 1, 20000) );
						break;
					case 5:
						dMapVal = GeoUtils.getRowNormedMatrix(GeoUtils.getInverseDistanceMatrix(samplesVal, gDist, 2, 20000) );
						dMapTrain = GeoUtils.getRowNormedMatrix(GeoUtils.getInverseDistanceMatrix(samplesTrain, gDist, 2, 20000) );
						break;
					case 6:
						dMapVal = GeoUtils.getRowNormedMatrix(GeoUtils.getInverseDistanceMatrix(samplesVal, gDist, 0, 5000) );
						dMapTrain = GeoUtils.getRowNormedMatrix(GeoUtils.getInverseDistanceMatrix(samplesTrain, gDist, 0, 5000) );
						break;
					default:	
						dMapVal = GeoUtils.getRowNormedMatrix( GeoUtils.listsToWeights( GeoUtils.getKNNs( samplesVal, gDist, 8, false ) ) );
						dMapTrain = GeoUtils.getRowNormedMatrix( GeoUtils.listsToWeights( GeoUtils.getKNNs( samplesTrain, gDist, 8, false ) ) );
				}
				System.gc();
			}
		}
		
		final List<HousepriceData> data = new ArrayList<HousepriceData>();
		for( int i = 0; i < maxRun; i++ )
			data.add( new HousepriceData( samples, desired ) );
		
		for( final int dmMode : new int[]{6,2} ) {
		for( HousepriceData hd : data )
			hd.setDistanceMatrices(dmMode);
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
			for( int l = 1; l <= nrNeurons; l++ )
				;//models.add( new Object[]{model.CNG,null,null,null,l} );
			
			for( final double alpha : new double[]{ 0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, /*0.5, 0.6, 0.7, 0.8, 0.9*/ } )
			for( final double beta :  new double[]{ 0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, /*0.5, 0.6, 0.7, 0.8, 0.9*/ } )
			for( final double gamma : new double[]{ 0.0, 0.01, 0.05, 0.1, 0.2, /*0.3, 0.4, /*0.5, 0.6, 0.7, 0.8, 0.9*/ } )
				models.add( new Object[]{model.WMNG, alpha, beta,gamma,null} );
			
			// good for all var
			//models.add( new Object[]{model.WMNG, 0.3, 0.0, 0.1,null} ); // knn8 and weight
			
			// good for no-ctx, knn8
			/*models.add( new Object[]{model.WMNG, 0.01, 0.05, 0.1,null} ); 
			models.add( new Object[]{model.WMNG, 0.3, 0.01, 0.1,null} ); 
			
			models.add( new Object[]{model.WMNG, 0.3, 0.0, 0.1,null} ); // good 
			models.add( new Object[]{model.WMNG, 0.1, 0.0, 0.1,null} ); // good*/
			
			// good for 1 var
			//models.add( new Object[]{model.WMNG, 0.1, 0.01, 0.2,null} );
			//models.add( new Object[]{model.WMNG, 0.01, 0.01, 0.2,null} );
												
			log.debug("models: "+models.size());
									
			for( final Object[] m : models ) {
			long time = System.currentTimeMillis();
			
			ExecutorService es = Executors.newFixedThreadPool(4);
			Map<HousepriceData,Future<List<double[]>>> futures = new HashMap<HousepriceData,Future<List<double[]>>>();
			
			for ( final HousepriceData hd : data ) {
				
				futures.put( hd, es.submit(new Callable<List<double[]>>() {

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
							double gamma = (double)m[3];
												
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
							for (int t = 0; t < T_MAX; t++) {
								int j = r.nextInt(samplesTrain.size());
								ng.train((double) t / T_MAX, samplesTrain.get(j), desiredTrain.get(j));
							}
														
							//ng.gamma = gamma;
							
							// ------------------------------------------
							
							final List<double[]> samplesVal = hd.samplesVal;
							
							Map<double[], Map<double[], Double>> dMapVal = hd.dMapVal;
							sorter.setWeightMatrix(dMapVal); // new weight-matrix
							
							bmuHist.clear();
							for (double[] d : samplesVal)
								bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));
							for( int i = 0; i < 60; i++ ) // train histMap
								for( double[] x : samplesVal )
									sorter.getBMU(x, neurons);
																						
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
							
							List<double[]> responseVal = new ArrayList<double[]>();
							for (double[] x : hd.samplesVal)
								responseVal.add(ng.present(x));	
							return responseVal;
						}
					}
				}));
			}
			es.shutdown();
			
			try {
				DescriptiveStatistics ds[] = null;
				for (Entry<HousepriceData, Future<List<double[]>>> ff : futures.entrySet() ) {
				
						List<double[]> response = ff.getValue().get();
						List<double[]> desiredVal = ff.getKey().desiredVal;
						double[] ee = new double[]{ Meuse.getRMSE(response, desiredVal) };
						
						if (ds == null) {
							ds = new DescriptiveStatistics[ee.length];
							for (int i = 0; i < ee.length; i++)
								ds[i] = new DescriptiveStatistics();
						}
						for (int i = 0; i < ee.length; i++)
							ds[i].addValue(ee[i]);
					
				}
				String s = dmMode+","+(model)m[0]+","+T_MAX+","+nrNeurons+","+nbInit+","+nbFinal+","+lr1Init+","+lr1Final+","+lr2Init+","+lr2Final+","+(m[1]==null ? "" : (double)m[1])+","+(m[2]==null ? "" : (double)m[2])+","+(m[3]==null ? "" : (double)m[3])+","+(m[4]==null ? "" : (int)m[4]);
				for (int i = 0; i < ds.length; i++)
					s += ","+ds[i].getMean()+","+ds[i].getVariance();
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
			log.debug("model took: "+(System.currentTimeMillis()-time)/1000.0+"s");
		}
		}
		log.debug("took: "+(System.currentTimeMillis()-timeAll)/1000.0/60.0+"m");
	}
	}
}
