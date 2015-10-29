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

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
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

import com.vividsolutions.jts.geom.Geometry;

public class NonstatClusterOptimize {

	private static Logger log = Logger.getLogger(NonstatClusterOptimize.class);

	public static void main(String[] args) {
		boolean firstWrite = true;
		final Random r = new Random();
		final DecimalFormat df = new DecimalFormat("00");

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/cng_llm/nonstatCluster.shp"),true);
		final List<double[]> samples = sdf.samples;
		final List<Geometry> geoms = sdf.geoms;
		final List<double[]> desired = new ArrayList<double[]>();

		for (double[] d : samples) {

			d[3] += r.nextDouble()*0.1-0.05; // add some noise
			desired.add(new double[] { d[3] });
		}

		final int[] fa = new int[] { 2 };
		final int[] ga = new int[] { 0, 1 };
		
		DataUtils.zScoreColumns(samples, fa);
		DataUtils.zScoreColumn(desired, 0); // should not be necessary

		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
				
		final Map<double[],Map<double[],Double>> rMap = GeoUtils.getInverseDistanceMatrix(samples, gDist, 1);
		GeoUtils.rowNormalizeMatrix(rMap);
		
		final Map<Integer,Set<double[]>> refColor = new HashMap<Integer,Set<double[]>>();
		for( double[] d : samples ) {
			int c = (int)d[5];
			if( !refColor.containsKey(c) )
				refColor.put(c, new HashSet<double[]>());
			refColor.get(c).add(d);
		}
		
		final Map<Integer,Set<double[]>> refClass = new HashMap<Integer,Set<double[]>>();
		for( double[] d : samples ) {
			int c = (int)d[4];
			if( !refClass.containsKey(c) )
				refClass.put(c, new HashSet<double[]>());
			refClass.get(c).add(d);
		}

		for( final int T_MAX : new int[]{ 40000 } )
			for( final int nrNeurons : new int[]{ 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,64 } )
			for( final double lInit : new double[]{ 3 })
			for( final double lFinal : new double[]{ 0.01 })	
			for( final double lr1Init : new double[]{ 0.7 })
			for( final double lr1Final : new double[]{ 0.01 })
			for( final double lr2Init : new double[]{ 0.5 })
			for( final double lr2Final : new double[]{ lr1Final })
			for( final LLMNG.mode mode : new LLMNG.mode[]{ LLMNG.mode.fritzke } )
			//for (int l = 5; l <= 5; l++ ) {
			for (int l : new int[]{ nrNeurons } ) {
			final int L = l;

			ExecutorService es = Executors.newFixedThreadPool(4);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (int run = 0; run < 32; run++) {

				futures.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {

						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < nrNeurons; i++) {
							double[] d = samples.get(r.nextInt(samples.size()));
							neurons.add(Arrays.copyOf(d, d.length));
						}

						ErrorSorter errorSorter = new ErrorSorter(samples, desired);
						DefaultSorter<double[]> gSorter = new DefaultSorter<>(gDist);
						Sorter<double[]> sorter = new KangasSorter<>(gSorter, errorSorter, L);

						DecayFunction nbRate = new PowerDecay(neurons.size()/lInit, lFinal);
						DecayFunction lrRate1 = new PowerDecay(lr1Init, lr1Final);
						DecayFunction lrRate2 = new PowerDecay(lr2Init, lr2Final);
						LLMNG ng = new LLMNG(neurons, 
								nbRate, lrRate1, 
								nbRate, lrRate2, 
								sorter, fa, 1 );
						errorSorter.setLLMNG(ng);
						ng.aMode = mode;
						
						for (int t = 0; t < T_MAX; t++) {
							int j = r.nextInt(samples.size());
							ng.train((double) t / T_MAX, samples.get(j), desired.get(j));
						}
												
						List<double[]> response = new ArrayList<double[]>();
						for (double[] x : samples)
							response.add(ng.present(x));					
						double mse =  Meuse.getMSE(response, desired);
											
						Map<double[],Double> errors = new HashMap<double[],Double>();
						for( int i = 0; i < response.size(); i++ )
							errors.put(samples.get(i), response.get(i)[0] - desired.get(i)[0] );
												
						//double[] moran1 = GeoUtils.getMoransIStatistics(rMap, errors);
						double[] moran = new double[]{GeoUtils.getMoransI(rMap, errors),0,0,0,0};
						//double[] moran1 = new double[]{0,0,0,0,0};
						
						int nrParams = nrNeurons * ( /*ga.length +*/ 2 * fa.length + 1 + 1 );  
						Map<double[],Set<double[]>> mapping = NGUtils.getBmuMapping(samples, neurons, sorter);
																													
						return new double[] { 
								mse,
								ClusterValidation.getNormalizedMutualInformation(mapping.values(), refColor.values()),
								ClusterValidation.getNormalizedMutualInformation(mapping.values(), refClass.values()),
								Math.abs( moran[0] ),
								moran[4],
								Meuse.getAIC(mse, nrParams, samples.size() ),
								Meuse.getAICc(mse, nrParams, samples.size() ),
								Meuse.getBIC(mse, nrParams, samples.size() )
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
				String fn = "output/resultNonstatCluster.csv";
				if( firstWrite ) {
					firstWrite = false;
					Files.write(Paths.get(fn), ("t_max,nrNeurons,l,lInit,lFinal,lr1Init,lr1Final,lr2Init,lr2Final,rmse,nmiColor,nmiClass,moran,pValue,aic,aicc,bic\n").getBytes());
				}
				String s = T_MAX+","+nrNeurons+","+l+","+lInit+","+lFinal+","+lr1Init+","+lr1Final+","+lr2Init+","+lr2Final+"";
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
}
