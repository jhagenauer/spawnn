package spawnn.utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
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

	// only used for local moran's i, TODO use knn/matrix-methods 
	private static double[][] getWeightMatrix(List<double[]> samples, Dist<double[]> gDist) {
		double threshold = 9999; // dists greater than th are cut off

		double max = Double.MIN_VALUE;
		for (int i = 0; i < samples.size(); i++) {
			for (int j = 0; j < samples.size(); j++) {
				double d = gDist.dist(samples.get(i), samples.get(j));
				if (d > max)
					max = d;
			}
		}

		// build inverse distance matrix
		double[][] w = new double[samples.size()][samples.size()];
		for (int i = 0; i < samples.size(); i++) {
			for (int j = 0; j < samples.size(); j++) {
				double d = gDist.dist(samples.get(i), samples.get(j));
				if (i == j || d > threshold)
					w[i][j] = 0;
				else {
					w[i][j] = 1 - 1 / d; // d/max;//max; // 1/d;
				}
			}
		}

		// k nearest neighbors
		/*
		 * for( int i = 0; i < w.length; i++ ) { List<Integer> nearest = new
		 * ArrayList<Integer>();
		 * 
		 * for( int k = 0; k < 5; k++ ) {
		 * 
		 * double m = Double.MAX_VALUE; int mIdx = -1;
		 * 
		 * for( int j = 0; j < w[i].length; j++ ) { if( i == j ||
		 * nearest.contains(j) ) continue;
		 * 
		 * if( w[i][j] < m ) { m = w[i][j]; mIdx = j; } } nearest.add(mIdx); }
		 * 
		 * for( int j = 0; j < w[i].length; j++ ) if( !nearest.contains(j) ) {
		 * w[i][j] = 0; w[j][i] = 0; } }
		 */

		// check for 0-rows/columns
		boolean f1 = false, f2 = false;
		for (int i = 0; i < w.length; i++) {
			for (int j = 0; j < w[i].length; j++) {
				if (w[i][j] != 0)
					f1 = true;
				if (w[j][i] != 0)
					f2 = true;
			}
		}

		if (!f1 || !f2)
			log.warn("Empty row/column in weight table! " + f1 + f2);

		/*
		 * row standardize For polygon features, you will almost always want to
		 * choose Row for the Standardization parameter. Row Standardization
		 * mitigates bias when the number of neighbors each feature has is a
		 * function of the aggregation scheme or sampling process, rather than
		 * reflecting the actual spatial distribution # of the variable you are
		 * analyzing.
		 */

		for (int i = 0; i < w.length; i++) {
			double sum = 0;
			for (int j = 0; j < w[i].length; j++)
				sum += w[i][j];
			for (int j = 0; j < w[i].length; j++)
				w[i][j] /= sum;
		}

		return w;
	}

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
	public static double getLocalMoransI(int idx, List<double[]> samples, Dist<double[]> gDist, int f) {

		double[][] w = getWeightMatrix(samples, gDist);

		// avergage
		double avg = 0;
		for (double[] d : samples)
			avg += d[f];
		avg /= samples.size();

		// first term denominator
		double ftd = 0;
		for (int i = 0; i < samples.size(); i++)
			ftd += Math.pow(samples.get(i)[f] - avg, 2);

		double s = 0;
		for (int i = 0; i < samples.size(); i++)
			s += w[idx][i] * (samples.get(i)[f] - avg);

		return ((samples.get(idx)[f] - avg) * s) / (ftd / samples.size());
	}
}
