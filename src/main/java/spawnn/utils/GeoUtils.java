package spawnn.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;

import cern.colt.Arrays;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;

public class GeoUtils {

	private static Logger log = Logger.getLogger(GeoUtils.class);
	
	public static void rowNormalizeMatrix( Map<double[],Map<double[],Double>> map ) {
		for( double[] a : map.keySet() ) {
			double sum = 0;
			for( double d : map.get(a).values() )
				sum += d;
						
			for( double[] b : new ArrayList<double[]>(map.get(a).keySet() ) )
				map.get(a).put(b, map.get(a).get(b)/sum );
		}
	}
	
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
	public static Map<double[],Map<double[],Double>> getKNearestMatrix( final Map<double[],Map<double[],Double>> invDistMatrix, int k ) {
		Map<double[],Map<double[],Double>> knnM = new HashMap<double[],Map<double[],Double>>();
		for( double[] a : invDistMatrix.keySet() ) {
			List<double[]> l = new ArrayList<double[]>( invDistMatrix.get(a).keySet() );
			Collections.sort(l, new Comparator<double[]>() {
				@Override
				public int compare(double[] o1, double[] o2) {
					return invDistMatrix.get(o1).get(o2).compareTo(invDistMatrix.get(o1).get(o2));
				}
			});			
			Map<double[],Double> m = new HashMap<double[],Double>();
			for( double[] d : l.subList(0, 1) )
				m.put(d, invDistMatrix.get(a).get(d) );
			knnM.put(a, m);
		}
		return knnM;
	}

	public static Map<double[], Map<double[], Double>> getInverseDistanceMatrix(Collection<double[]> samples, Dist<double[]> gDist, double pow) {
		Map<double[], Map<double[], Double>> r = new HashMap<double[], Map<double[], Double>>();
		
		double minDist = -1;
		for (double[] a : samples) {
			Map<double[], Double> m = new HashMap<double[], Double>();

			for (double[] b : samples) {
				if (a == b)
					continue;
				
				double dist = gDist.dist(a, b);
				if( dist == 0 ) {
					if( minDist < 0  ) { // only calc/show message once
						minDist = Double.POSITIVE_INFINITY;
						for (double[] aa : samples) { 
							for (double[] bb : samples) {
								if( aa == bb )
									continue;
								double d = gDist.dist(aa, bb);
								if( d > 0 && d < minDist )
									minDist = d;
							}
						}		
						log.warn("Identical points present. Setting dist to "+minDist);
					}
					dist = minDist;
				}
				m.put(b, 1.0 / Math.pow(dist, pow));
			}
			r.put(a, m);
		}
		return r;
	}
	
	public static Map<double[], Map<double[], Double>> listsToWeights( Map<double[], List<double[]>> connectMap ) {
		Map<double[], Map<double[], Double>> r = new HashMap<double[], Map<double[], Double>>();
		for( double[] a : connectMap.keySet() ) {
			r.put(a, new HashMap<double[],Double>() );
			for( double[] nb : connectMap.get(a) )
				r.get(a).put(nb, 1.0 );
		}
		return r;
	}
	
	public static Map<double[], List<double[]>> getKNNs(final List<double[]> samples, final Dist<double[]> gDist, int k) {
		Map<double[], List<double[]>> r = new HashMap<double[], List<double[]>>();
		for (final double[] x : samples) {
			
			List<double[]> l = new ArrayList<double[]>( samples );
			Collections.sort(l, new Comparator<double[]>() {
				@Override
				public int compare(double[] o1, double[] o2) {
					return Double.compare(gDist.dist(x, o1),gDist.dist(x, o2));
				}
			});		
			r.put(x, l.subList(0, k));
		}
		return r;
	}
	
	@Deprecated
	public static Map<double[], List<double[]>> getContiguityMap(List<double[]> samples, List<Geometry> geoms, boolean rook ) {
		Map<double[], List<double[]>> r = new HashMap<double[], List<double[]>>();
		for( int i = 0; i < samples.size(); i++ ) {
			Geometry a = geoms.get(i);
			List<double[]> l = new ArrayList<double[]>();
			for( int j = 0; j < samples.size(); j++ ) {
				Geometry b = geoms.get(j);
				if( !rook ) { // queen
					if( a.touches(b) || a.intersects(b) )
						l.add( samples.get(j));
				} else { // rook
					if( a.intersection(b).getCoordinates().length > 0 ) // SLOW
						l.add( samples.get(j));
				}
			}
			r.put(samples.get(i), l);
		}
		return r;
	}
	
	public static Map<double[], List<double[]>> getContiguityMap(List<double[]> samples, List<Geometry> geoms, boolean rookAdjacency, boolean includeIdentity ) {
		Map<double[], List<double[]>> r = new HashMap<double[], List<double[]>>();
		for( int i = 0; i < samples.size(); i++ ) {
			Geometry a = geoms.get(i);
			List<double[]> l = new ArrayList<double[]>();
			for( int j = 0; j < samples.size(); j++ ) {
				Geometry b = geoms.get(j);
				if( !includeIdentity && a == b )
					continue;				
				if( !rookAdjacency ) { // queen
					if( a.touches(b) || a.intersects(b) )
						l.add( samples.get(j));
				} else { // rook
					if( a.intersection(b).getCoordinates().length > 0 ) // SLOW
						l.add( samples.get(j));
				}
			}
			r.put(samples.get(i), l);
		}
		return r;
	}

	// morans ------------------------------>>>
	
	public static double getMoransI(Map<double[], Map<double[], Double>> dMap, Map<double[],Double> values ) {		
		double n = values.size();
		double mean = 0;
		for (double[] d : values.keySet() )
			mean += values.get(d) / n;

		// first term denominator
		double ftd = 0;
		for (double[] d : values.keySet() )
			ftd += Math.pow(values.get(d) - mean, 2);

		// sec term numerator
		double stn = 0;
		for (double[] d1 : dMap.keySet())
			for (double[] d2 : dMap.get(d1).keySet())		
				stn += dMap.get(d1).get(d2) * (values.get(d1) - mean) * (values.get(d2) - mean);
		
		// sec term denominator
		double std = 0;
		for (Map<double[], Double> m : dMap.values())
			for (double d : m.values()) 
				std += d;
						
		return (n / ftd) * (stn / std);
	}
	
	public static double[] getMoransIStatistics( Map<double[], Map<double[], Double>> dMap, Map<double[],Double> values ) {
		double n = values.size();
		double moran = getMoransI(dMap, values);
		
		double E_I = -1.0/(n-1);
		
		// calculate variance, from wikipedia
		double s1 = 0;
		for( double[] i : dMap.keySet() ) 
			for( double[] j : dMap.keySet() )
				if( i != j )
					s1 += Math.pow(dMap.get(i).get(j)+dMap.get(j).get(i), 2);
		s1 *= 0.5;
		
		double s2 = 0;
		for( double[] i : dMap.keySet() ) {
			double s = 0;
			for( double[] j : dMap.keySet() )
				if( i != j )
					s += dMap.get(i).get(j);
			for( double[] j : dMap.keySet() )
				if( i != j )
					s += dMap.get(j).get(i);
			s2 += Math.pow(s, 2);
		}
		
		double m = 0;
		for( double d : values.values() )
			m += d;
		m /= values.size();
		
		double s3 = 0;
		for( double d : values.values() )
			s3 += Math.pow(d-m,4);
		s3 /= n;
		
		double nom = 0;
		for( double d : values.values() )
			nom += Math.pow(d-m,2);
		s3 /= Math.pow( nom/n, 2);
		
		double sumWij = 0;
		for( double[] i : dMap.keySet() ) 
			for( double[] j : dMap.keySet() )
				if( i != j )
					sumWij += dMap.get(i).get(j);
		
		double s4 = (Math.pow(n,2)-3*n+3)*s1 - s2*n + 3*Math.pow(sumWij,2);	
		double s5 = (Math.pow(n,2)-n)*s1 - 2*n*s2 + 6*Math.pow(sumWij,2);
		
		double Var_I = ( (n * s4 - s3 * s5) / ((n-1)*(n-2)*(n-3)*Math.pow(sumWij, 2)) ) - Math.pow(E_I, 2);	
		double zScore = (moran - E_I )/Math.sqrt(Var_I);
		NormalDistribution nd = new NormalDistribution();
										
		return new double[]{ 
				moran, 
				E_I,
				Var_I,
				zScore,
				2*nd.density(-Math.abs(zScore)) 
			};
	}
	
	public static double[] getMoransIStatisticsMonteCarlo( Map<double[], Map<double[], Double>> dMap, Map<double[],Double> values, int reps ) {
		double moran = getMoransI(dMap, values);
		double n = values.size();
		
		DescriptiveStatistics ds = new DescriptiveStatistics();
		for( int i = 0; i < reps; i++ ) {
			
			// permute
			List<Double> l = new ArrayList<Double>(values.values());
			Collections.shuffle(l);
			Map<double[],Double> m = new HashMap<double[],Double>();
			int j = 0;
			for( double[] d : values.keySet() )
				m.put(d, l.get(j++));
			
			double permMoran = getMoransI(dMap, m);
			ds.addValue(permMoran);
		}
		
		TDistribution td = new TDistribution(n-1); // ????
		
		double tStatistic = ( moran - ds.getMean() ) / Math.sqrt(ds.getVariance() ); 
		return new double[]{ 
				moran,
				ds.getMean(),
				ds.getVariance(),
				tStatistic, // ?????
				2*td.density(-Math.abs(tStatistic)), // p-Value????
			};
	}

	@Deprecated
	public static double getMoransI(Map<double[], Map<double[], Double>> dMap, int fa) {
		Set<double[]> samples = new HashSet<double[]>(dMap.keySet());
		for (double[] d : dMap.keySet()) 
			samples.addAll(dMap.get(d).keySet());
		
		Map<double[],Double> values = new HashMap<double[],Double>();
		for( double[] d : samples )
			values.put(d, d[fa]);
		return getMoransI(dMap, values);
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
	
	public static void main(String[] args) {
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("data/ozone.csv"), new int[]{2,3}, new int[]{}, true);
		List<double[]> samples = sdf.samples;
		Dist<double[]> gDist = new EuclideanDist(new int[]{2,3});
		
		Map<double[], Map<double[], Double>> m1 = getInverseDistanceMatrix(samples, gDist, 1);
		Map<double[],Double> values = new HashMap<double[],Double>();
		for( double[] d : samples )
			values.put(d, d[1]);
		log.debug( "Inv, 1, norm: "+getMoransI( getRowNormedMatrix(m1), values) ); 
		log.debug(Arrays.toString(getMoransIStatistics(m1, values )));
		log.debug(Arrays.toString(getMoransIStatisticsMonteCarlo(m1, values, 2000000 )));
	}

	public static <T> void writeDistMatrixKeyValue(Map<T, Map<T, Double>> dMap, List<T> samples, File fn) {
		Map<T,Integer> idxMap = new HashMap<T,Integer>();
		for( int i = 0; i < samples.size(); i++ )
			idxMap.put(samples.get(i), i);
		try {
			FileWriter fw = new FileWriter(fn);
			fw.write("id1,id2,dist\n");
			for (Entry<T, Map<T, Double>> e1 : dMap.entrySet()) {
				int a = idxMap.get(e1.getKey());
				for (Entry<T, Double> e2 : e1.getValue().entrySet())
					fw.write(a + "," + idxMap.get(e2.getKey()) + "," + e2.getValue() + "\n");
			}
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static <T> Map<T, Map<T, Double>> readDistMatrixKeyValue(List<T> samples, File fn) throws NumberFormatException, IOException, FileNotFoundException {
		Map<T, Map<T, Double>> distMatrix = new HashMap<T, Map<T, Double>>();
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(fn));
			String line = br.readLine(); // ignore first line by reading but not using
			while ((line = br.readLine()) != null) {
	
				String[] s = line.split(",");
	
				T a = samples.get(Integer.parseInt(s[0]));
				T b = samples.get(Integer.parseInt(s[1]));
	
				if (!distMatrix.containsKey(a))
					distMatrix.put(a, new HashMap<T, Double>());
	
				distMatrix.get(a).put(b, Double.parseDouble(s[2]));
			}
		} finally {
			try {
				if( br != null )
					br.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return distMatrix;
	}
}
