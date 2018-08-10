package context.cng;

import java.io.File;
import java.util.ArrayList;
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

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D_Map;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class TestPA {
				
	public static void main( String args[] ) {
		final Random r = new Random();
		
		final int T_MAX = 100000;				
		//final List<Geometry> geoms = DataUtil.readGeometriesFromShapeFile(new File("data/marco/dat1/pgo_regression_transform.shp"));
		final List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/marco/dat1/pgo_regression_transform.shp"), new int[] {}, false);

		int[] fa = { 6, 7, 8, 9, 10, 11, 14 };
		int[] ga = { 4, 5 };
		
		final Dist eDist = new EuclideanDist();
		final Dist geoDist = new EuclideanDist(ga );
		final Dist fDist = new EuclideanDist(fa );
		
		DataUtils.normalizeColumns( samples, fa );
		DataUtils.normalizeGeoColumns( samples, ga ); 
				
		int runs = 100;
		int threads = 16;
				
		// weighted k-Means
		/*{
		ExecutorService es = Executors.newFixedThreadPool(threads);
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
		
		final int min_k = 0, max_k = 100;
		double[][] qe = new double[runs][max_k-min_k+1];
		double[][] ge = new double[runs][max_k-min_k+1];
		double[][] time = new double[runs][max_k-min_k+1];
					
		for( double weight = min_k; weight <= max_k; weight++ ) {
			final double WEIGHT = weight;
						
			for( int i = 0; i < runs; i++ ) {
				final int I = i;
				
				futures.add( es.submit( new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
			
						Map<Dist,Double> wMap = new HashMap<Dist,Double>();
						wMap.put(fDist, (double)WEIGHT/max_k);
						wMap.put(geoDist, 1-(double)WEIGHT/max_k);
						
						Dist wDist = new WeightedDist(wMap);
						
						long ti = System.currentTimeMillis();
						Map<double[],Set<double[]>> m = DataUtil.kMeans(samples, 5, wDist);
						double time = (double)(System.currentTimeMillis()-ti);
						
						double qe = DataUtil.getQuantError( m, fDist );
						double ge = DataUtil.getQuantError( m, geoDist );
						
						return new double[]{ WEIGHT, I, time, qe, ge };
					}
				}));
			
			}
		}
		es.shutdown();
		
		for( Future<double[]> f : futures ) {
			try {
				double[] d = f.get();
				int k = (int)d[0];
				int i = (int)d[1];
				time[i][k-min_k] = d[2];
				qe[i][k-min_k] = d[3];
				ge[i][k-min_k] = d[4];	
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
		DataUtil.saveTab( "output/test_vq_kmeans_qe.csv", qe);
		DataUtil.saveTab( "output/test_vq_kmeans_ge.csv", ge);
		DataUtil.saveTab( "output/test_vq_kmeans_time.csv", time);
		}*/
		
		// geosom
		{
		ExecutorService es = Executors.newFixedThreadPool(threads);
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
			
		int min_k = 0, max_k = 4;
		double[][] qe = new double[runs][max_k-min_k+1];
		double[][] ge = new double[runs][max_k-min_k+1];
		double[][] time = new double[runs][max_k-min_k+1];
		double[][] ke = new double[runs][max_k-min_k+1];
				
		for( int k = min_k; k <= max_k; k++ ) {			
			for( int i = 0; i < runs; i++ ) {
					
				final int K = k, I = i;
								
				futures.add( es.submit( new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
						Grid2D_Map<double[]> grid = new Grid2DHex<double[]>(5, 1 );
						//grid.initLinear(samples, true);
						spawnn.som.bmu.BmuGetter<double[]> bmuGetter = new spawnn.som.bmu.KangasBmuGetter( geoDist, fDist, K);
						SOM som = new SOM( new GaussKernel(grid.getMaxDist()), new LinearDecay(0.5,0.0), grid, bmuGetter );
						
						long ti = System.currentTimeMillis();
						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(r.nextInt(samples.size() ) );
							som.train( (double)t/T_MAX, x );
						}
						
						double time = (double)(System.currentTimeMillis()-ti);						
						
						Map<double[],Set<double[]>> cluster = new HashMap<double[],Set<double[]>>();
						for( double[] d : samples ) {
							double[] bmu = grid.getPrototypeAt( bmuGetter.getBmuPos(d, grid ) );
							if( !cluster.containsKey(bmu) )
								cluster.put(bmu, new HashSet<double[]>() );
							cluster.get(bmu).add(d);
						}
																
						double qe = DataUtils.getMeanQuantizationError( cluster, fDist );
						double ge = DataUtils.getMeanQuantizationError( cluster, geoDist );
						
						return new double[]{ K, I, time, qe, ge, -1 };
					}
				}));
			}
		}
		es.shutdown();
		
		for( Future<double[]> f : futures ) {
			try {
				double[] d = f.get();
				int k = (int)d[0];
				int i = (int)d[1];
				time[i][k-min_k] = d[2];
				qe[i][k-min_k] = d[3];
				ge[i][k-min_k] = d[4];
				ke[i][k-min_k] = d[5];
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}

		DataUtils.writeTab( "output/test_pa_geosom_time.csv", time);
		DataUtils.writeTab( "output/test_pa_geosom_qe.csv", qe);
		DataUtils.writeTab( "output/test_pa_geosom_ge.csv", ge);
		DataUtils.writeTab( "output/test_pa_geosom_ke.csv", ke);
		}
		
		// cng
		{
		ExecutorService es = Executors.newFixedThreadPool(threads);
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
			
		final int min_k = 1, max_k = 5;
		double[][] qe = new double[runs][max_k-min_k+1];
		double[][] ge = new double[runs][max_k-min_k+1];
		double[][] time = new double[runs][max_k-min_k+1];
		
		for( int k = min_k; k <= max_k; k++ ) {			
			for( int i = 0; i < runs; i++ ) {
				
				final int K = k, I = i;
			
				futures.add( es.submit( new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
						Sorter bmuGetter = new KangasSorter( geoDist, fDist, K );
						//NeuralGas ng = new NeuralGas(max_k, 1, 0.0165, 0.4899, 0.00591, samples.get(0).length, bmuGetter );
						NG ng = new NG(max_k, max_k/2, 0.01, 0.5, 0.005, samples.get(0).length, bmuGetter );
						
						long ti = System.currentTimeMillis();
						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(r.nextInt(samples.size() ) );
							ng.train( (double)t/T_MAX, x );
						}
						double time = (double)(System.currentTimeMillis()-ti);
						
						Map<double[],Set<double[]>> cluster = NGUtils.getBmuMapping(samples, ng.getNeurons(), bmuGetter);
												
						double qe = DataUtils.getMeanQuantizationError( cluster, fDist );
						double ge = DataUtils.getMeanQuantizationError( cluster, geoDist );
						
						return new double[]{ K, I, time, qe, ge };
					}
				}));
			}
		}	
		es.shutdown();
		
		for( Future<double[]> f : futures ) {
			try {
				double[] d = f.get();
				int k = (int)d[0];
				int i = (int)d[1];
				time[i][k-min_k] = d[2];
				qe[i][k-min_k] = d[3];
				ge[i][k-min_k] = d[4];
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
					
		DataUtils.writeTab( "output/test_pa_cng_time.csv", time);
		DataUtils.writeTab( "output/test_pa_cng_qe.csv", qe);
		DataUtils.writeTab( "output/test_pa_cng_ge.csv", ge);
		}
	}
}
