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

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import llm.LLMNG;
import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.sorter.SorterWMC;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class BostonTest_LOO {
	
	private static Logger log = Logger.getLogger(BostonTest_LOO.class);
	
	enum model {WMNG, NG, NG_LAG, WNG_LAG, LM};
		
	public static void main(String[] args) {
		long timeAll = System.currentTimeMillis();
		final Random r = new Random();
		
		int maxRun = 1;
		int threads = 4;
		
		String fn = "output/resultBostonTest_llo.csv";
		fn = fn.replaceAll(" ","");
		try {
			Files.write(Paths.get(fn), ("vars,t_max,nrNeurons,nbInit,nbFinal,lr1Init,lr1Final,lr2Init,lr2Final,model,alpha,beta,rmse,r2\n").getBytes());
		} catch (IOException e) {
			e.printStackTrace();
		}
					
		// was passiert wenn wir NICHT transformieren?
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/boston/boston_tracts.shp"), true);
		DataUtils.transform(sdf.samples, new int[]{5,12,13,17}, Transform.log);
		DataUtils.transform(sdf.samples, new int[]{9,10}, Transform.pow2);
		DataUtils.transform(sdf.samples, Transform.zScore);
		
		for( int[] vars : new int[][]{
				/*new int[]{6},
				new int[]{7},
				new int[]{8},
				new int[]{9},
				new int[]{10},
				new int[]{11},
				new int[]{12},
				new int[]{13},
				new int[]{14},
				new int[]{15},
				new int[]{16},*/
				new int[]{17},
				/*new int[]{18},*/
		} ) {

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
				
		Map<double[],Map<double[],Double>> dMap = GeoUtils.getRowNormedMatrix( GeoUtils.contiguityMapToDistanceMap(GeoUtils.getContiguityMap(samples, sdf.geoms, false, false)));
						
		final int[] fa = new int[samples.get(0).length];
		for( int i = 0; i < fa.length; i++ )
			fa[i] = i;
							
		final Dist<double[]> fDist = new EuclideanDist(fa);
				
		final List<TrainingDataLOO> data = new ArrayList<TrainingDataLOO>();
		for( int i = 0; i < samples.size(); i++ ) // full
			data.add( new TrainingDataLOO( i, samples, desired, dMap ) );
					
		for( final int T_MAX : new int[]{ 120000 } )	
		for( final int nrNeurons : new int[]{ 16 } ) 		
		for( final double nbInit : new double[]{ (double)nrNeurons*2.0/3.0 })
		for( final double nbFinal : new double[]{ 1.0 })	
		for( final double lr1Init : new double[]{ 0.6 }) 
		for( final double lr1Final : new double[]{ 0.001 })
		for( final double lr2Init : new double[]{ 0.1 })
		for( final double lr2Final : new double[]{ 0.001 })
		{			
			List<Object[]> models = new ArrayList<Object[]>();
			//models.add( new Object[]{model.NG_LAG,null,null,null,null} );
			models.add(new Object[] { model.NG_LAG, 1, null });
			models.add(new Object[] { model.NG_LAG, 2, null });
			
			/*for (double alpha = 0.0; alpha <= 1.0; alpha = (double)Math.round( (alpha+0.05) * 100000) / 100000 )
				models.add(new Object[] { model.WNG_LAG, alpha, null });*/
			
			for (double alpha = 0.0; alpha <= 1.0; alpha = (double)Math.round( (alpha+0.05) * 100000) / 100000 )
			for (double beta = 0; beta <= 1.0; beta = (double)Math.round( (beta+0.05) * 100000) / 100000 )
				models.add( new Object[]{model.WMNG, alpha, beta} );
																		
			log.debug("models: "+models.size());
									
			for( final Object[] m : models ) {
			long time = System.currentTimeMillis();
			
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for( int run = 0; run < maxRun; run++ ) {
			
			ExecutorService es = Executors.newFixedThreadPool(threads);
			Map<TrainingDataLOO,Future<double[]>> futures = new HashMap<TrainingDataLOO,Future<double[]>>();
			
			for ( final TrainingDataLOO hd : data ) {
				
				futures.put( hd, es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
						
						List<double[]> samplesTrain = hd.samplesTrain;
						List<double[]> desiredTrain = hd.desiredTrain;
						Map<double[], Map<double[], Double>> dMapTrain = hd.dMapTrain;
												
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
								
							SorterWMC sorter = new SorterWMC(bmuHist, dMapTrain, fDist, alpha, beta);
							ContextNG_LLM ng = new ContextNG_LLM(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, fa, 1);
							for (int t = 0; t < T_MAX; t++) {
								int j = r.nextInt(samplesTrain.size());
								ng.train((double) t / T_MAX, samplesTrain.get(j), desiredTrain.get(j));
							}														
														
							sorter.setWeightMatrix(hd.dMapVal); // full weight-matrix
							return ng.present(hd.sampleVal);	
						} else if( m[0] == model.NG_LAG ){
							int lag = (int) m[1];
							
							List<double[]> lagedSamplesTrain = GeoUtils.getLagedSamples(samplesTrain, dMapTrain, lag);
							
							List<double[]> neurons = new ArrayList<double[]>();
							for (int i = 0; i < nrNeurons; i++) {
								double[] d = lagedSamplesTrain.get(r.nextInt(lagedSamplesTrain.size()));
								neurons.add(Arrays.copyOf(d, d.length));
							}
							
							int[] nfa = null;
							if( lag == 1 ) {
								nfa = new int[fa.length*2];
								for( int i = 0; i < fa.length; i++ ) {
									nfa[i] = fa[i];
									nfa[i+fa.length] = fa[i]+samplesTrain.get(0).length;
								}
							} else if( lag == 2 ) {
								nfa = new int[fa.length*3];
								for( int i = 0; i < fa.length; i++ ) {
									nfa[i] = fa[i];
									nfa[i+fa.length] = fa[i]+samplesTrain.get(0).length;
									nfa[i+2*fa.length] = fa[i]+2*samplesTrain.get(0).length;
								}
							} 
														
							Sorter<double[]> sorter = new DefaultSorter<>( new EuclideanDist(nfa));
							LLMNG ng = new LLMNG(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, nfa, 1);

							for (int t = 0; t < T_MAX; t++) {
								int j = r.nextInt(lagedSamplesTrain.size());
								ng.train((double) t / T_MAX, lagedSamplesTrain.get(j), desiredTrain.get(j));
							}

							List<double[]> l = new ArrayList<double[]>();
							l.add(hd.sampleVal);
							List<double[]> lagedSamplesVal = GeoUtils.getLagedSamples(l, hd.dMapVal,lag);
							return ng.present(lagedSamplesVal.get(0));
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
							return ng.present(hd.sampleVal);
						} else if( m[0] == model.WNG_LAG ) { 
							double alpha = (double)m[1];
							
							List<double[]> lagedSamplesTrain = GeoUtils.getLagedSamples(samplesTrain, dMapTrain);
							
							List<double[]> neurons = new ArrayList<double[]>();
							for (int i = 0; i < nrNeurons; i++) {
								double[] d = lagedSamplesTrain.get(r.nextInt(lagedSamplesTrain.size()));
								neurons.add(Arrays.copyOf(d, d.length));
							}
							
							int[] faLag = new int[fa.length];
							for( int i = 0; i < fa.length; i++ )
								faLag[i] = samplesTrain.get(0).length + fa[i];
							
							int[] nfa = new int[fa.length*2];
							for( int i = 0; i < fa.length; i++ ) {
								nfa[i] = fa[i];
								nfa[fa.length + i] = faLag[i];
							}
							
							Map<Dist<double[]>,Double> m = new HashMap<Dist<double[]>,Double>();
							m.put(fDist, 1.0-alpha);
							m.put(new EuclideanDist(faLag), alpha);
							
							Sorter<double[]> sorter = new DefaultSorter<>( new WeightedDist<>(m));

							LLMNG ng = new LLMNG(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, nfa, 1);

							for (int t = 0; t < T_MAX; t++) {
								int j = r.nextInt(lagedSamplesTrain.size());
								ng.train((double) t / T_MAX, lagedSamplesTrain.get(j), desiredTrain.get(j));
							}

							List<double[]> l = new ArrayList<double[]>();
							l.add(hd.sampleVal);
							List<double[]> lagedSamplesVal = GeoUtils.getLagedSamples(l, hd.dMapVal);
							return ng.present(lagedSamplesVal.get(0));
							
						} else {
							return null;
						}
					}
				}));
			}
			es.shutdown();
			
			try {
				
				List<double[]> responseVal = new ArrayList<double[]>();
				List<double[]> desiredVal = new ArrayList<double[]>();
				List<double[]> samplesVal = new ArrayList<double[]>();
				
				Map<double[],Double> residuals = new HashMap<double[],Double>();
				for (Entry<TrainingDataLOO, Future<double[]>> ff : futures.entrySet() ) {
					responseVal.add( ff.getValue().get() );
					samplesVal.add( ff.getKey().sampleVal );
					desiredVal.add( ff.getKey().desiredVal );
					residuals.put(ff.getKey().sampleVal, ff.getValue().get()[0] - ff.getKey().desiredVal[0] );
				}
				
				ds.addValue( Meuse.getRMSE(responseVal, desiredVal)  );
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
			
		}
			
		log.debug("model took: "+(System.currentTimeMillis()-time)/1000.0+"s");
		try {
			String s = vNames+","+T_MAX+","+nrNeurons+","+nbInit+","+nbFinal+","+lr1Init+","+lr1Final+","+lr2Init+","+lr2Final+","+Arrays.toString(m).replaceAll("\\[", "").replaceAll("\\]", "");
			s += ","+ds.getMean()+"\n";
			Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
			System.out.print(s);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		}
		}
		}
		
		log.debug("took: "+(System.currentTimeMillis()-timeAll)/1000.0/60.0+"m");
		log.debug(fn+" written.");
	}
}
