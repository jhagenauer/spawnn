package spawnn.utils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import spawnn.dist.Dist;

public class ClusterValidation {
	
	public static double getNormalizedMutualInformation(Collection<Set<double[]>> u1, Collection<Set<double[]>> v1) {
		return DataUtils.getNormalizedMutualInformation(u1, v1);
	}

	// works identically to R-cluster package
	public static double getSilhouetteCoefficient(Map<double[], Set<double[]>> cluster, Dist<double[]> dist) {
		int sum = 0;
		double sil = 0;
		for (double[] curCentroid : cluster.keySet()) {
			for (double[] a : cluster.get(curCentroid)) {

				// get average distances to each cluster
				Map<double[], Double> avgDistMap = new HashMap<double[], Double>();
				for (double[] key : cluster.keySet()) {
					double avgDist = 0;
					for (double[] b : cluster.get(key))
						avgDist += dist.dist(a, b);
					if (cluster.get(key).contains(a))
						avgDistMap.put(key, avgDist / (cluster.get(key).size() - 1));
					else
						avgDistMap.put(key, avgDist / cluster.get(key).size());
				}

				// get distA and distB
				double distA = avgDistMap.get(curCentroid); // avg dist to
															// current cluster
				double distB = Double.MAX_VALUE;
				for (double[] key : cluster.keySet())
					// get cluster with best avgDist
					if (key != curCentroid && avgDistMap.get(key) < distB)
						distB = avgDistMap.get(key);

				if (distA != 0)
					sil += (distB - distA) / Math.max(distA, distB);

				sum++;
			}
		}
		return sil / sum;
	}

	public static double getDaviesBouldinIndex(Map<double[], Set<double[]>> clusters, Dist<double[]> dist) {
		double sum = 0;

		for (double[] c1Center : clusters.keySet()) {
			Set<double[]> c1 = clusters.get(c1Center);

			double max = Double.MIN_VALUE;
			for (double[] c2Center : clusters.keySet()) {
				Set<double[]> c2 = clusters.get(c2Center);

				if (c1.equals(c2))
					continue;

				double si = 0;
				for (double[] d : c1)
					si += dist.dist(d, c1Center);
				si /= c1.size();

				double sj = 0;
				for (double[] d : c2)
					sj += dist.dist(d, c2Center);
				sj /= c2.size();

				double r = (si + sj) / dist.dist(c1Center, c2Center);

				if (r > max)
					max = r;
			}
			sum += max;
		}
		return sum / clusters.size();
	}

	// aka compactness
	public static double getWithinClusterSumOfSuqares(Collection<Set<double[]>> c, Dist<double[]> dist) {
		return DataUtils.getWithinSumOfSuqares(c, dist);
	}

	// separation
	public static double getBetweenClusterSumOfSuqares(Collection<Set<double[]>> c, Dist<double[]> dist) {
		int length = c.iterator().next().iterator().next().length;
		double[] meanOfMeans = new double[length];
		
		int numSamples = 0;
		for( Set<double[]> s : c ) {
			double[] m = DataUtils.getMean(s);
			for( int i = 0; i < m.length; i++ )
				meanOfMeans[i] += m[i]/c.size();
			numSamples += s.size();
		}
		
		double v = 0;
		for( Set<double[]> s : c ) 
			v += ((double)s.size()/numSamples) * Math.pow( dist.dist( DataUtils.getMean(s), meanOfMeans), 2);
		return v;
	}

	// dunn
	public static double getDunnIndex(Collection<Set<double[]>> c, Dist<double[]> dist ) {
		double dunn = Double.POSITIVE_INFINITY;
		
		double maxDia = 0;
		for( Set<double[]> s : c ) 
			for( double[] a : s )
				for( double[] b : s )
					maxDia = Math.max( dist.dist(a, b), maxDia);
		
		for( Set<double[]> k : c )
			for( Set<double[]> l : c ) {
				if( k == l )
					continue;
				
				double minDist = Double.POSITIVE_INFINITY;
				for( double[] a : k )
					for( double[] b : l )
						minDist = Math.min( dist.dist(a,b), minDist);
				
				dunn = Math.min( dunn, minDist/maxDia);
			}
		return dunn;
	}

	// connectivity
	public static double getConnectivity(Collection<Set<double[]>> c, Dist<double[]> dist, int numNBs) {
		List<double[]> samples = new ArrayList<double[]>();
		for( Set<double[]> s : c )
			samples.addAll(s);
		
		double conn = 0;
		for( Set<double[]> s : c ) {
			for( double[] d : s ) {
				for( int j = 1; j <= numNBs; j++ ) {
					double[] nb = getNearestNeighbor(samples, d, dist, j);
					if( s.contains(nb) )
						conn += 1.0/j;
				}
			}
		}
		return conn;
	}
	
	private static double[] getNearestNeighbor( List<double[]> samples, double[] d, Dist<double[]> dist, int k) {
		List<double[]> nns = new ArrayList<double[]>();
		nns.add(d); // 0th nearest neighbor
				
		while( nns.size() - 1 < k  ) {
			double[] nearest = null;
			for( double[] x : samples ) {
				if( nns.contains(x) )
					continue;
				
				if( nearest == null || dist.dist(x, d) < dist.dist(nearest,d) )
					nearest = x;
			}
			nns.add(nearest);
		}
		
		return nns.get(nns.size()-1);
	}

}
 