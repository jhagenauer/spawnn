package gwr.ga;

import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

import spawnn.utils.GeoUtils.GWKernel;

public abstract class GWRCostCalculator implements CostCalculator<GWRIndividual> {
	
	protected List<double[]> samples;
	protected int[] fa, ga;
	protected int ta, minBW;
	protected  GWKernel kernel;
	
	GWRCostCalculator( List<double[]> samples, int[] fa, int[] ga, int ta, GWKernel kernel, int minBW) {
		this.samples = samples;
		this.fa = fa;
		this.ga = ga;
		this.ta = ta;
		this.minBW = minBW;
		this.kernel = kernel;		
	}
	
	protected int getBandwidthAt( GWRIndividual ind, int i ) {
		return (int)Math.min( Math.max( minBW, Math.round( ind.getBandwidth().get(i) ) ), ind.getBandwidth().size() );
	}
	
	protected double[] getKthLargest(List<double[]> samples, int k, Comparator<double[]> c) {
		PriorityQueue<double[]> q = new PriorityQueue<double[]>(k, c);
		for (double[] d : samples) {
			q.offer(d);
			if (q.size() > k)
				q.poll();
		}
		return q.peek();
	}
}