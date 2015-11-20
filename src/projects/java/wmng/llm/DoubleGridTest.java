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
import spawnn.som.grid.Grid2D;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;

public class DoubleGridTest {
	
	private static Logger log = Logger.getLogger(DoubleGridTest.class);
	
	enum model {WMNG,CNG};
		
	public static void main(String[] args) {
		boolean firstWrite = true;
		final Random r = new Random();
								
		int maxRun = 16;
		final int[] fa = new int[]{2};
		final int[] ga = new int[] { 0, 1 };
				
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		long time = System.currentTimeMillis();
		class GridData {
			public GridData(Grid2D<double[]> grid) {
				this.dMap = GeoUtils.getRowNormedMatrix(GeoUtils.listsToWeights(GeoUtils.getNeighborsFromGrid(grid)));
				
				for( double[] d : grid.getPrototypes() ) {
					samples.add(d);
					desired.add(new double[]{d[d.length-1]});
				}
				DataUtils.zScoreColumns(samples, fa);
				DataUtils.zScoreColumn(desired, 0);
			}
			Map<double[], Map<double[], Double>> dMap;
			List<double[]> samples = new ArrayList<double[]>();
			List<double[]> desired = new ArrayList<double[]>();
		}
		
		final Map<Integer,GridData> dataTrain = new HashMap<Integer,GridData>();
		final Map<Integer,GridData> dataVal = new HashMap<Integer,GridData>();
		for( int i = 0; i < maxRun; i++ ) {
			dataTrain.put( i, new GridData( DoubleGrid2DUtils.createDoubleGrid(50, 50, 3, true) ) );
			dataVal.put(   i, new GridData( DoubleGrid2DUtils.createDoubleGrid(50, 50, 3, true) ) );
		}
		log.debug("Data create took: "+(System.currentTimeMillis()-time)/1000.0+"s");
		time = System.currentTimeMillis();
										
		for( final int T_MAX : new int[]{ 40000 } )	
		for( final int nrNeurons : new int[]{ 8 } ) 		
		for( final double nbInit : new double[]{ (double)nrNeurons*2.0/3.0 })
		for( final double nbFinal : new double[]{ 0.1 })	
		for( final double lr1Init : new double[]{ 0.6 }) 
		for( final double lr1Final : new double[]{ 0.01 })
		for( final double lr2Init : new double[]{ 0.1, 0.2, 0.4 })
		for( final double lr2Final : new double[]{ 0.01 })
		{			
			List<Object[]> models = new ArrayList<Object[]>();
			for( final boolean useContext : new boolean[]{ false, true } )
				for( final double alpha : new double[]{ 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.0 } )
					for( final double beta : new double[]{ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 } )
						models.add( new Object[]{model.WMNG, useContext, alpha, beta,null} );
			
			for( int l = 1; l <= nrNeurons; l++ )
				models.add( new Object[]{model.CNG,null,null,null,l} );
			
			for( final Object[] m : models ) {
			
			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
			
			for (int run = 0; run < maxRun; run++) {
				final int RUN = run;

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
						
						List<double[]> samplesTrain = dataTrain.get(RUN).samples;
						List<double[]> desiredTrain = dataTrain.get(RUN).desired;
												
						DecayFunction nbRate = new PowerDecay(nbInit, nbFinal);
						DecayFunction lrRate1 = new PowerDecay(lr1Init, lr1Final);
						DecayFunction lrRate2 = new PowerDecay(lr2Init, lr2Final);
						
						if( m[0] == model.WMNG ) { // WMNG + LLM
							boolean useContext = (boolean)m[1];
							double alpha = (double)m[2];
							double beta = (double)m[3];
												
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
	
							Map<double[], Map<double[], Double>> dMapTrain = dataTrain.get(RUN).dMap;
							SorterWMC sorter = new SorterWMC(bmuHist, dMapTrain, fDist, alpha, beta);
													
							ContextNG_LLM ng = new ContextNG_LLM(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, fa, 1, useContext);
							for (int t = 0; t < T_MAX; t++) {
								int j = r.nextInt(samplesTrain.size());
								ng.train((double) t / T_MAX, samplesTrain.get(j), desiredTrain.get(j));
							}
							ng.alpha = alpha;
							
							// ------------------------------------------
							
							final List<double[]> samplesVal = dataVal.get(RUN).samples;
							final List<double[]> desiredVal = dataVal.get(RUN).desired;
														
							bmuHist.clear();
							for (double[] d : samplesVal)
								bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));
									
							Map<double[], Map<double[], Double>> dMapVal = dataVal.get(RUN).dMap;
							sorter.setWeightMatrix(dMapVal); // new weight-matrix
							for( int i = 0; i < 100; i++ ) // train histMap
								for( double[] x : samplesVal )
									sorter.sort(x, neurons);
													
							List<double[]> responseVal = new ArrayList<double[]>();
							for (double[] x : samplesVal)
								responseVal.add(ng.present(x));				
																										
							return new double[] { Meuse.getRMSE(responseVal, desiredVal) };
						} else { // CNG+LLM
							List<double[]> neurons = new ArrayList<double[]>();
							for (int i = 0; i < nrNeurons; i++) {
								double[] d = samplesTrain.get(r.nextInt(samplesTrain.size()));
								neurons.add(Arrays.copyOf(d, d.length));
							}

							Sorter<double[]> secSorter = new DefaultSorter<>(fDist);
							DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
							Sorter<double[]> sorter = new KangasSorter<>(gSorter, secSorter, (int)m[4] );
							
							LLMNG ng = new LLMNG(neurons, 
									nbRate, lrRate1, 
									nbRate, lrRate2, 
									sorter, fa, 1 );
							
							for (int t = 0; t < T_MAX; t++) {
								int j = r.nextInt(samplesTrain.size());
								ng.train((double) t / T_MAX, samplesTrain.get(j), desiredTrain.get(j));
							}
							
							final List<double[]> samplesVal = dataVal.get(RUN).samples;
							final List<double[]> desiredVal = dataVal.get(RUN).desired;
							
							List<double[]> responseVal = new ArrayList<double[]>();
							for (double[] x : samplesVal)
								responseVal.add(ng.present(x));	
							return new double[] { Meuse.getRMSE(responseVal, desiredVal) };
						}
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
				String fn = "output/resultContextNG_LLM.csv";
				if( firstWrite ) {
					firstWrite = false;
					Files.write(Paths.get(fn), ("model,t_max,nrNeurons,nbInit,nbFinal,lr1Init,lr1Final,lr2Init,lr2Final,useContext,alpha,beta,l,rmse\n").getBytes());
				}
				String s = (model)m[0]+","+T_MAX+","+nrNeurons+","+nbInit+","+nbFinal+","+lr1Init+","+lr1Final+","+lr2Init+","+lr2Final+","+(m[1]==null ? "" : (boolean)m[1])+","+(m[2]==null ? "" : (double)m[2])+","+(m[3]==null ? "" : (double)m[3])+","+(m[4]==null ? "" : (int)m[4]);
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
		log.debug("took: "+(System.currentTimeMillis()-time)/1000.0/60.0+" min");	
	}
}
