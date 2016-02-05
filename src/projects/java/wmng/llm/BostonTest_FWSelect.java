package wmng.llm;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
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
import spawnn.ng.sorter.Sorter;
import spawnn.ng.sorter.SorterWMC;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.transform;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class BostonTest_FWSelect {
	
	private static Logger log = Logger.getLogger(BostonTest_FWSelect.class);
	
	enum model {WMNG, NG, NG_LAG};
		
	public static void main(String[] args) {
		long timeAll = System.currentTimeMillis();
		final Random r = new Random();
		
		int threads = 4;
		int maxRun = 16; 
				
		model selectBy = model.WMNG;
		String fn = "output/resultBostonTest_FWSelect_"+selectBy+"_"+maxRun+".csv";
		fn = fn.replaceAll(" ","");
		try {
			Files.write(Paths.get(fn), ("vars,model,alpha,beta,rmse\n").getBytes());
		} catch (IOException e) {
			e.printStackTrace();
		}
			
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/boston/boston_tracts.shp"), true);
		DataUtils.transform(sdf.samples, new int[]{5,12,13,17}, transform.log); // without transform performance is bad
		DataUtils.transform(sdf.samples, new int[]{9,10}, transform.pow2);
		DataUtils.transform(sdf.samples, transform.zScore);
		
		int[] allVars = new int[]{7,8,9,11,12,13,14,15,16,18,/*17,10,6*/};
				
		Set<Integer> unusedVars = new HashSet<Integer>();
		for( int i : allVars )
			unusedVars.add(i);
		List<Integer> curVars = new ArrayList<Integer>();
		curVars.add(17);
		curVars.add(10);
		curVars.add(6);
				
		while( !unusedVars.isEmpty() ) {
					
		Integer bestVar = null;
		Map<Object[],Double> rmseMapbestVar = null;
		double best = Double.MAX_VALUE;
		
		for( Integer v : unusedVars ) {
			curVars.add(v);
											
			final List<double[]> samples = new ArrayList<double[]>();
			final List<double[]> desired = new ArrayList<double[]>();
			for (double[] d : sdf.samples) {
				double[] nd = new double[curVars.size()];
				for( int i = 0; i < curVars.size(); i++ )
					nd[i] = d[curVars.get(i)];
				samples.add(nd);
				desired.add(new double[]{d[5]});
			}
							
			Map<double[],Map<double[],Double>> dMap = GeoUtils.getRowNormedMatrix( GeoUtils.listsToWeights(GeoUtils.getContiguityMap(samples, sdf.geoms, false, false)));
			final TrainingData hd = new TrainingData( samples, desired, dMap );
			
			final int[] fa = new int[samples.get(0).length];
			for( int i = 0; i < fa.length; i++ )
				fa[i] = i;								
			final Dist<double[]> fDist = new EuclideanDist(fa);		
			
			Map<Object[],Double> rmseMap = new HashMap<Object[],Double>();
						
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
				//models.add( new Object[]{model.NG,null,null,null} );
				//models.add( new Object[]{model.NG_LAG,null,null} );
				
				for (double alpha = 0.0; alpha <= 1; alpha = (double)Math.round( (alpha+0.05) * 100000) / 100000 )
				for (double beta = 0; beta <= 1; beta = (double)Math.round( (beta+0.05) * 100000) / 100000 )
					models.add( new Object[]{model.WMNG, alpha, beta} );															
										
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
					DescriptiveStatistics ds = new DescriptiveStatistics();
					for (Future<List<double[]>> ff : futures ) {
							List<double[]> responseVal = ff.get();
							List<double[]> desiredVal = hd.desiredVal;
							ds.addValue(Meuse.getRMSE(responseVal, desiredVal));	
					}
					rmseMap.put(m, ds.getMean());
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
				log.debug(curVars+","+Arrays.toString(m)+" took: "+(System.currentTimeMillis()-time)/1000.0+"s");
			}	
			}
			for( Entry<Object[],Double> e : rmseMap.entrySet() ) {
				if( (model)e.getKey()[0] == selectBy && e.getValue() < best ) {
					bestVar = v;
					best = e.getValue();
					rmseMapbestVar = rmseMap;
				}
			}
			curVars.remove(v);
		}
		curVars.add(bestVar);
		unusedVars.remove(bestVar);
		log.debug(bestVar+","+best);
		
		for( Entry<Object[],Double> e : rmseMapbestVar.entrySet() ) {
			String s = sdf.names.get(curVars.get(0));
			for( int i = 1; i < curVars.size(); i++ ) 
				s += ";"+sdf.names.get(curVars.get(i));
			s +=","+Arrays.toString(e.getKey()).replaceAll("\\[", "").replaceAll("\\]", "")+","+e.getValue()+"\n";
			System.out.print(s);
			try {
				Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
			} catch (IOException e1) {
				e1.printStackTrace();
			}
		}						
		}
		log.debug("took: "+(System.currentTimeMillis()-timeAll)/1000.0/60.0+"m");
		log.debug(fn+" written.");
	}
}
