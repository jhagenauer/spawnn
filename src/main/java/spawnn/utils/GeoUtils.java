package spawnn.utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;

public class GeoUtils {

	private static Logger log = Logger.getLogger(GeoUtils.class);
	
	public static Map<double[],Map<double[],Double>> getRowNormedMatrix( Map<double[],Map<double[],Double>> map ) {
		Map<double[],Map<double[],Double>> normedMatrix = new HashMap<double[],Map<double[],Double>>();

		for( double[] a : map.keySet() ) {
			double sum = 0;
			for( double d : map.get(a).values() )
				sum += d;
			
			Map<double[], Double> n = new HashMap<double[], Double>();
			for( double[] b : map.get(a).keySet() )
				n.put(b,map.get(a).get(b)/sum );
			normedMatrix.put(a, n);
		}
		return normedMatrix;
	}
	
	// useful to strip large distance-matrices from not relevant entries (????)
	public static Map<double[],Map<double[],Double>> getKNearestMatrix( Map<double[],Map<double[],Double>> invDistMatrix, int k ) {
		Map<double[],Map<double[],Double>> knnM = new HashMap<double[],Map<double[],Double>>();
		for( double[] a : invDistMatrix.keySet() ) {
			Map<double[],Double> m = new HashMap<double[],Double>();
			while( m.size() < k ) {
				double[] maxB = null;
				for( double[] b : invDistMatrix.get(a).keySet() ) {
					if( !m.containsKey(b) && ( maxB == null || invDistMatrix.get(a).get(maxB) < invDistMatrix.get(a).get(b) ) )
						maxB = b;
				}
				m.put(maxB, invDistMatrix.get(a).get(maxB));
			}
			knnM.put(a, m);
		}
		return knnM;
	}

	public static Map<double[], Map<double[], Double>> getInverseDistanceMatrix(List<double[]> samples, Dist<double[]> gDist, int pow) {
		Map<double[], Map<double[], Double>> r = new HashMap<double[], Map<double[], Double>>();
		
		double minDist = Double.POSITIVE_INFINITY;
		for (double[] a : samples) { 
			for (double[] b : samples) {
				if( a == b )
					continue;
				double d = gDist.dist(a, b);
				if( d > 0 && d < minDist )
					minDist = d;
			}
		}
		
		boolean warn = true;
		
		for (double[] a : samples) {
			Map<double[], Double> m = new HashMap<double[], Double>();

			for (double[] b : samples) {
				if (a == b)
					continue;
				
				double dist = gDist.dist(a, b);
				if( dist == 0 ) {
					if( warn ) {
						log.warn("Identical points present. Setting dist to "+minDist);
						warn = false;
					}
					dist = minDist;
				}
				m.put(b, 1.0 / Math.pow(dist, pow));
			}
			r.put(a, m);
		}
		return r;
	}
	
	public static Map<double[], Map<double[], Double>> knnsToWeights( Map<double[], List<double[]>> knns ) {
		Map<double[], Map<double[], Double>> r = new HashMap<double[], Map<double[], Double>>();
		for( double[] a : knns.keySet() ) {
			r.put(a, new HashMap<double[],Double>() );
			for( double[] nb : knns.get(a) )
				r.get(a).put(nb, 1.0/knns.get(a).size() );
		}
		return r;
	}

	public static Map<double[], List<double[]>> getKNNs(List<double[]> samples, Dist<double[]> gDist, int k) {
		Map<double[], List<double[]>> r = new HashMap<double[], List<double[]>>();
		for (double[] x : samples) {
			List<double[]> sub = new ArrayList<double[]>();
			while (sub.size() <= k) {
				double[] minD = null;
				for (double[] d : samples)
					if (!sub.contains(d) && (minD == null || gDist.dist(d, x) < gDist.dist(minD, x)))
						minD = d;
				sub.add(minD);
			}
			r.put(x, sub);
		}
		return r;
	}

	// morans ------------------------------>>>

	public static double getMoransI(Map<double[], Map<double[], Double>> dMap, int fa) {
		Set<double[]> samples = new HashSet<double[]>();
		for (double[] d : dMap.keySet()) {
			samples.add(d);
			for (double[] d2 : dMap.get(d).keySet())
				samples.add(d2);
		}

		double n = samples.size();
		double mean = 0;
		for (double[] d : samples)
			mean += d[fa] / n;

		// first term denominator
		double ftd = 0;
		for (double[] d : samples)
			ftd += Math.pow(d[fa] - mean, 2);

		// sec term numerator
		double stn = 0;
		for (double[] d1 : dMap.keySet())
			for (double[] d2 : dMap.get(d1).keySet())		
				stn += dMap.get(d1).get(d2) * (d1[fa] - mean) * (d2[fa] - mean);
		
		// sec term denominator
		double std = 0;
		for (Map<double[], Double> m : dMap.values())
			for (double d : m.values()) 
				std += d;
				
		return (n / ftd) * (stn / std);
	}

	// only univariate at the moment
	public static List<Double> getLocalMoransI(List<double[]> samples, Map<double[], Map<double[], Double>> dMap, int fa) {
		double mean = 0;
		for (double[] d : samples)
			mean += d[fa];
		mean /= samples.size();
		
		double m2 = 0;
		for( double[] d : samples )
			m2 += Math.pow(d[fa] - mean,2);
		m2 /= samples.size();
		
		List<Double> lisa = new ArrayList<Double>();
		for( double[] d : samples ) {
			double ii = 0;
			
			Map<double[],Double> nbs = dMap.get(d);
			for( double[] nb : nbs.keySet() ) 
				ii += nbs.get(nb) * (nb[fa] - mean);
			
			ii *= (d[fa] - mean)/m2;
			lisa.add(ii);
						
		}
		return lisa;
	}
}
