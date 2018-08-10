package context.cng;

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
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.utils.DataUtils;

/* Beispiel schiefe ebene, unterschiedliche autocorrelation, in der diagonalen weniger, in den anderen ecken hoch
 */

public class TestVQ {
				
	public static void main( String args[] ) {
		final Random r = new Random();
		
		final int T_MAX = 100000;				
		// final List<double[]> samples = DataUtil.readSamplesFromShapeFile(new File("data/cng/test2c.shp"), new int[]{}, true);
		
		final List<double[]> samples = new ArrayList<double[]>();
		/* 1000, x+y: 0.01 - 0.035
		 * 1000, x*y:0.01 - 0.035
		 */
		for( int i = 0; i < 1000; i++ ) {
			double x = r.nextDouble();
			double y = r.nextDouble();
			samples.add( new double[]{ x, y, x*y, x+y, Math.pow(x,y) } );
		}
							
		final int[] fa = { 2 };
		final int[] ga = new int[]{0,1};
		
		final Dist<double[]> eDist = new EuclideanDist();
		final Dist<double[]> geoDist = new EuclideanDist(ga );
		final Dist<double[]> fDist = new EuclideanDist(fa );
		
		DataUtils.normalizeColumns( samples, fa );
		DataUtils.normalizeGeoColumns( samples, ga );
					
		int runs = 10;
		int threads = 4;
		
		// weighted k-Means
		/*{
		ExecutorService es = Executors.newFixedThreadPool(threads);
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
		
		final int min_k = 0, max_k = 10;
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
						wMap.put(fDist, WEIGHT/max_k);
						wMap.put(geoDist, 1.0 - WEIGHT/max_k);
						
						Dist wDist = new WeightedDist(wMap);
						
						long ti = System.currentTimeMillis();
						Map<double[],Set<double[]>> m = DataUtil.kMeans(samples, 25, wDist);
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
			
		int min_k = 0, max_k = 6;
		double[][] qe = new double[runs][max_k-min_k+1];
		double[][] ge = new double[runs][max_k-min_k+1];
		double[][] te = new double[runs][max_k-min_k+1];
		double[][] time = new double[runs][max_k-min_k+1];
				
		for( int k = min_k; k <= max_k; k++ ) {			
			for( int i = 0; i < runs; i++ ) {
					
				final int K = k, I = i;
								
				futures.add( es.submit( new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
						Grid2D<double[]> grid = new Grid2DHex<double[]>(5, 5);
																		
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
						double te = 0; //SomUtils.getTopoError(grid, bmuGetter, samples);
						
						return new double[]{ K, I, time, qe, ge, te };
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
				te[i][k-min_k] = d[5];
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}

		DataUtils.writeTab( "output/test_vq_geosom_time.csv", time);
		DataUtils.writeTab( "output/test_vq_geosom_qe.csv", qe);
		DataUtils.writeTab( "output/test_vq_geosom_ge.csv", ge);
		DataUtils.writeTab( "output/test_vq_geosom_te.csv", te);
		}
				
		// cng
		{
		ExecutorService es = Executors.newFixedThreadPool(threads);
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
			
		final int min_k = 1, max_k = 25;
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
						NG ng = new NG(25, 25/2, 0.01, 0.5, 0.005, samples.get(0).length, bmuGetter );
						
						long ti = System.currentTimeMillis();
						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(r.nextInt(samples.size() ) );
							ng.train( (double)t/T_MAX, x );
						}
						double time = (double)(System.currentTimeMillis()-ti);
						
						Map<double[],Set<double[]>> cluster = new HashMap<double[],Set<double[]>>();
						for( double[] w : ng.getNeurons() )
							cluster.put( w, new HashSet<double[]>() );
						for( double[] d : samples ) {
							bmuGetter.sort(d, ng.getNeurons() );
							double[] bmu = ng.getNeurons().get(0);
							cluster.get(bmu).add(d);
						}
												
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
					
		DataUtils.writeTab( "output/test_vq_cng_time.csv", time);
		DataUtils.writeTab( "output/test_vq_cng_qe.csv", qe);
		DataUtils.writeTab( "output/test_vq_cng_ge.csv", ge);
		}
	}
}
