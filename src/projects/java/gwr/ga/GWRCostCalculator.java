package gwr.ga;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.GeoUtils.GWKernel;

public abstract class GWRCostCalculator implements CostCalculator<GWRIndividual> {
	
	protected List<double[]> samples;
	protected int[] fa, ga;
	protected int ta;
	protected  GWKernel kernel;
	protected Dist<double[]> gDist;
	protected boolean adaptive;
	protected double minBw;
	
	GWRCostCalculator( List<double[]> samples, int[] fa, int[] ga, int ta, GWKernel kernel, boolean adaptive, double minBw ) {
		this.samples = samples;
		this.fa = fa;
		this.ga = ga;
		this.gDist = new EuclideanDist(ga);
		this.ta = ta;
		this.kernel = kernel;
		this.adaptive = adaptive;
		this.minBw = minBw;
	}
		
	private double[] getKthLargest(List<double[]> samples, int k, Comparator<double[]> c) {
		PriorityQueue<double[]> q = new PriorityQueue<double[]>(k, c);
		for (double[] d : samples) {
			q.offer(d);
			if (q.size() > k)
				q.poll();
		}
		return q.peek();
	}

	Map<double[],Map<Integer,Double>> adaptiveBwCache = new HashMap<double[],Map<Integer,Double>>();
 	
	public Map<double[],Double> getSpatialBandwidth(GWRIndividual ind) {
		Map<double[],Double> bandwidth = new HashMap<>();
		for (int i = 0; i < samples.size(); i++) {
			double[] a = samples.get(i);	
			if( adaptive ) {
				int k = Math.max( (int)minBw, Math.min( samples.size(), (int)Math.round( ind.getGeneAt(i) ) ) );			
				
				synchronized( adaptiveBwCache ) {
					if( !adaptiveBwCache.containsKey(a) )
						adaptiveBwCache.put(a, new HashMap<Integer,Double>() );
					if( !adaptiveBwCache.get(a).containsKey(k) ) {
						double[] b = getKthLargest(samples, k, new Comparator<double[]>() {
							@Override
							public int compare(double[] o1, double[] o2) {
								return -Double.compare(gDist.dist(o1, a), gDist.dist(o2, a));
							}
						});
						adaptiveBwCache.get(a).put(k,gDist.dist(a, b));
					}	
					bandwidth.put( a, adaptiveBwCache.get(a).get(k) );
				}
			} else {
				bandwidth.put( a, Math.max( minBw, ind.getGeneAt(i) ) );
			}
		}		
		return bandwidth;
	}
}