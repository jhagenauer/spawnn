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
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.utils.DataUtils;

public class Optimize {
				
	public static void main( String args[] ) {
		final Random r = new Random();
		
		final int T_MAX = 100000;				
		//final List<double[]> samples = DataUtil.readSamplesFromShapeFile(new File("data/marco/dat1/pgo_regression_transform.shp"), new int[] {}, false);
		//int[] fa = { 6, 7, 8, 9, 10, 11, 14 };
		//int[] ga = { 4, 5 };
		
		final List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/cng/test2c.shp"), new int[]{}, true);
		
		final int[] fa = { 2 };
		final int[] ga = new int[]{0,1};
		
		final Dist eDist = new EuclideanDist();
		final Dist geoDist = new EuclideanDist(ga );
		final Dist fDist = new EuclideanDist(fa );
		
		DataUtils.normalizeColumns( samples, fa );
		DataUtils.normalizeGeoColumns( samples, ga ); 
				
		int runs = 25;
		int threads = 16;
					
		// cng
		{
		double best = Double.POSITIVE_INFINITY;
			
		while( true ) {
			
			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
			
			final double ri = 5+r.nextInt(20);
			final double rf = r.nextDouble();
				
			final double ei = r.nextDouble();
			final double ef = r.nextDouble()*ei;
				
			for( int i = 0; i < runs; i++ ) {
						
				futures.add( es.submit( new Callable<double[]>() {
		
					@Override
					public double[] call() throws Exception {
						Sorter bmuGetter = new DefaultSorter(fDist);
															
						NG ng = new NG(25, ri, rf, ei, ef, samples.get(0).length, bmuGetter );
							
						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(r.nextInt(samples.size() ) );
							ng.train( (double)t/T_MAX, x );
						}
							
						Map<double[],Set<double[]>> cluster = new HashMap<double[],Set<double[]>>();
						for( double[] w : ng.getNeurons() )
							cluster.put( w, new HashSet<double[]>() );
						for( double[] d : samples ) {
							bmuGetter.sort(d, ng.getNeurons() );
							double[] bmu = ng.getNeurons().get(0);
							cluster.get(bmu).add(d);
						}
											
						return new double[]{DataUtils.getMeanQuantizationError( cluster, fDist ) };
					}
				}));
			}
						
			es.shutdown();
				
			double avgQe = 0;
			for( Future<double[]> f : futures ) {
				try {
					double[] d = f.get();
					avgQe += (double)d[0];			
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
			avgQe /= runs; 
						
			if( avgQe < best ) {
				best = avgQe;
				System.out.println("avgQe: "+avgQe);
				System.out.println(ri+":"+rf+":"+ei+":"+ef);
			}
			
		}			
	}
	}
}
