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

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D_Map;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.ClusterValidation;
import spawnn.utils.DataUtils;

public class TestCluster {
	
	private static Logger log = Logger.getLogger(TestCluster.class);
	
	/* Erzeuge x cluster, und schaue, ob GeoNG diese findet. Vergleich zu GeoSOM auch.
	 */
			
	public static void main( String args[] ) {
				
		final List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/cng/test2a_nonoise.shp"), new int[] {}, true);
		//final List<double[]> samples = DataUtil.readSamplesFromShapeFile(new File("data/cng/test2a.shp"), new int[] {}, true);
		//final List<double[]> samples = DataUtil.readSamplesFromShapeFile(new File("data/cng_var_test/100regions.shp"), new int[] {}, true);
		
				
		final Map<Integer,Set<double[]>> classes = new HashMap<Integer,Set<double[]>>();
		for( double[] d : samples ) {
			int c = (int)d[3];
			if( !classes.containsKey(c) )
				classes.put(c, new HashSet<double[]>() );
			classes.get(c).add(d);
		}
		
		final int numCluster = classes.size();
		
		final int[] fa = {2};
		final int[] ga = new int[]{0,1};
		
		final Dist<double[]> eDist = new EuclideanDist();
		final Dist<double[]> geoDist = new EuclideanDist(ga );
		final Dist<double[]> fDist = new EuclideanDist(fa );
		
		DataUtils.normalizeColumns(samples, fa );
		DataUtils.normalizeGeoColumns(samples, ga );
							
		final Random r = new Random();
		final int T_MAX = 100000;
					
		int runs = 10;
		int threads = 4;
		
		// weighted k-Means
		/*{
		log.debug("kmeans");
		ExecutorService es = Executors.newFixedThreadPool(threads);
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
		
		final int min_k = 0, max_k = 200;
		double[][] qe = new double[runs][max_k-min_k+1];
		double[][] ge = new double[runs][max_k-min_k+1];
		double[][] nmi = new double[runs][max_k-min_k+1];
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
						Map<double[],Set<double[]>> m = DataUtil.kMeans(samples, numCluster, wDist);
						double time = (double)(System.currentTimeMillis()-ti);
						
						double qe = DataUtil.getQuantError( m, fDist );
						double ge = DataUtil.getQuantError( m, geoDist );
						double nmi = DataUtil.getNormalizedMutualInformation(m.values(), classes.values() );
						
						return new double[]{ WEIGHT, I, time, qe, ge, nmi };
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
				nmi[i][k-min_k] = d[5];		
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
		DataUtil.saveTab( "output/test_cluster_kmeans_qe.csv", qe);
		DataUtil.saveTab( "output/test_cluster_kmeans_ge.csv", ge);
		DataUtil.saveTab( "output/test_cluster_kmeans_nmi.csv", nmi);
		DataUtil.saveTab( "output/test_cluster_kmeans_time.csv", time);
		}*/
		
		// geo som
		{ 
		log.debug("geosom");
		ExecutorService es = Executors.newFixedThreadPool(threads);
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
		
		int min_k = 0, max_k = 8;
		double[][] qe = new double[runs][max_k-min_k+1];
		double[][] ge = new double[runs][max_k-min_k+1];
		double[][] nmi = new double[runs][max_k-min_k+1];
		double[][] time = new double[runs][max_k-min_k+1];
		double[][] ke = new double[runs][max_k-min_k+1];
				
		for( int k = min_k; k <= max_k; k++ ) {			
			for( int i = 0; i < runs; i++ ) {
				
				final int K = k, I = i;
				
				futures.add( es.submit( new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
						Grid2D_Map<double[]> grid = new Grid2DHex<double[]>(5, 5 );
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
						double nmi = ClusterValidation.getNormalizedMutualInformation(cluster.values(), classes.values() );
																		
						return new double[]{ K, I, time, qe, ge, nmi, -1 };
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
				nmi[i][k-min_k] = d[5];	
				ke[i][k-min_k] = d[6];
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
		
		DataUtils.writeTab( "output/test_cluster_geosom_qe.csv", qe);
		DataUtils.writeTab( "output/test_cluster_geosom_ge.csv", ge);
		DataUtils.writeTab( "output/test_cluster_geosom_nmi.csv", nmi);
		DataUtils.writeTab( "output/test_cluster_geosom_time.csv", time);
		DataUtils.writeTab( "output/test_cluster_geosom_ke.csv", ke);
		}
		
		// cng
		{
		log.debug("cng");
		ExecutorService es = Executors.newFixedThreadPool(threads);
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
			
		final int min_k = 1, max_k = numCluster;
		double[][] qe = new double[runs][max_k-min_k+1];
		double[][] ge = new double[runs][max_k-min_k+1];
		double[][] nmi = new double[runs][max_k-min_k+1];
		double[][] time = new double[runs][max_k-min_k+1];
		
		for( int k = min_k; k <= max_k; k++ ) {
			for( int i = 0; i < runs; i++ ) {
				
				final int K = k, I = i;
				
								
				futures.add( es.submit( new Callable<double[]>() {
					@Override
					public double[] call() throws Exception {
						Sorter bmuGetter = new KangasSorter( geoDist, fDist, K );
						NG ng = new NG(numCluster, numCluster/2, 0.01, 0.5, 0.005, samples.get(0).length, bmuGetter );
						
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
						double nmi = ClusterValidation.getNormalizedMutualInformation(cluster.values(), classes.values() );
						
						return new double[]{ K, I, time, qe, ge, nmi };
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
				nmi[i][k-min_k] = d[5];		
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}

		DataUtils.writeTab( "output/test_cluster_cng_qe.csv", qe);
		DataUtils.writeTab( "output/test_cluster_cng_ge.csv", ge);
		DataUtils.writeTab( "output/test_cluster_cng_nmi.csv", nmi);
		DataUtils.writeTab( "output/test_cluster_cng_time.csv", time);
		}
	}
}
