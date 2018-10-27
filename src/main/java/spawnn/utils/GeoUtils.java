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
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;

import cern.colt.Arrays;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.grid.Grid;
import spawnn.som.grid.GridPos;

public class GeoUtils {

	private static Logger log = Logger.getLogger(GeoUtils.class);

	public static List<double[]> getLagedSamples(List<double[]> samples, Map<double[], Map<double[], Double>> dMap) {
		List<double[]> ns = new ArrayList<double[]>();
		for (double[] d : samples) {
			double[] lag = new double[d.length];
			for (Entry<double[], Double> nb : dMap.get(d).entrySet())
				for (int i = 0; i < lag.length; i++)
					lag[i] += nb.getValue() * nb.getKey()[i];

			// append
			double[] nd = new double[d.length * 2];
			for (int i = 0; i < d.length; i++) {
				nd[i] = d[i];
				nd[i + d.length] = lag[i];
			}
			ns.add(nd);
		}
		return ns;
	}

	public static List<double[]> getLagedSamples(List<double[]> samples, Map<double[], Map<double[], Double>> dMap,
			int lag) {
		List<double[]> ns = new ArrayList<double[]>();
		for (double[] d : samples) {
			double[] nd = new double[d.length * (lag + 1)];
			for (int i = 0; i < d.length; i++)
				nd[i] = d[i];

			for (int j = 1; j <= lag; j++) {
				double[] l = getLagedSample(d, dMap, j);
				for (int i = 0; i < d.length; i++)
					nd[i + j * d.length] = l[i];
			}
			ns.add(nd);
		}
		return ns;
	}

	public static double[] getLagedSample(double[] x, Map<double[], Map<double[], Double>> dMap, int lag) {
		double[] l = new double[x.length];
		if (lag == 1) {
			for (Entry<double[], Double> e : dMap.get(x).entrySet())
				for (int i = 0; i < l.length; i++)
					l[i] += e.getKey()[i] * e.getValue();
		} else if (lag == 2) {
			for (Entry<double[], Double> e1 : dMap.get(x).entrySet())
				for (Entry<double[], Double> e2 : dMap.get(e1.getKey()).entrySet())
					for (int i = 0; i < l.length; i++)
						l[i] += e2.getKey()[i] * e1.getValue() * e2.getValue();
		} else {
			throw new RuntimeException("Not implemented yet!");
		}
		return l;
	}

	public static Map<double[], List<double[]>> getNeighborsFromGrid(Grid<double[]> grid) {
		Map<double[], List<double[]>> m = new HashMap<double[], List<double[]>>();
		for (GridPos p : grid.getPositions()) {
			List<double[]> l = new ArrayList<double[]>();
			for (GridPos nb : grid.getNeighbours(p))
				l.add(grid.getPrototypeAt(nb));
			m.put(grid.getPrototypeAt(p), l);
		}
		return m;
	}

	public static void rowNormalizeMatrix(Map<double[], Map<double[], Double>> map) {
		for (double[] a : map.keySet()) {
			double sum = 0;
			for (double d : map.get(a).values())
				sum += d;

			for (double[] b : new ArrayList<double[]>(map.get(a).keySet()))
				map.get(a).put(b, map.get(a).get(b) / sum);
		}
	}

	public static Map<double[], Map<double[], Double>> getRowNormedMatrix(Map<double[], Map<double[], Double>> map) {
		Map<double[], Map<double[], Double>> normedMatrix = new HashMap<double[], Map<double[], Double>>();

		for (double[] a : map.keySet()) {
			double sum = 0;
			for (double d : map.get(a).values())
				sum += d;

			Map<double[], Double> n = new HashMap<double[], Double>();
			for (double[] b : map.get(a).keySet())
				n.put(b, map.get(a).get(b) / sum);
			normedMatrix.put(a, n);
		}
		return normedMatrix;
	}

	// useful to strip large distance-matrices from not relevant(??) entries
	public static Map<double[], Map<double[], Double>> getKNearestFromMatrix(
			final Map<double[], Map<double[], Double>> invDistMatrix, int k) {
		Map<double[], Map<double[], Double>> knnM = new HashMap<double[], Map<double[], Double>>();
		for (final double[] a : invDistMatrix.keySet()) {
			PriorityQueue<double[]> pq = new PriorityQueue<>(invDistMatrix.get(a).size(), new Comparator<double[]>() {
				@Override
				public int compare(double[] o1, double[] o2) {
					return invDistMatrix.get(a).get(o1).compareTo(invDistMatrix.get(a).get(o2));
				}
			});
			pq.addAll(invDistMatrix.get(a).keySet());

			Map<double[], Double> sub = new HashMap<double[], Double>();
			while (sub.size() < k) {
				double[] d = pq.poll();
				sub.put(d, invDistMatrix.get(a).get(d));
			}
			knnM.put(a, sub);
		}
		return knnM;
	}

	public static Map<double[], Map<double[], Double>> getInverseDistanceMatrix(Map<double[], Map<double[], Double>> m,
			double pow) {
		Map<double[], Map<double[], Double>> nm = new HashMap<double[], Map<double[], Double>>();
		for (Entry<double[], Map<double[], Double>> e1 : m.entrySet()) {
			Map<double[], Double> na = new HashMap<>();
			for (Entry<double[], Double> e2 : e1.getValue().entrySet()) {
				assert e2.getValue() > 0;
				na.put(e2.getKey(), Math.pow(1.0 / e2.getValue(), pow));
			}
			nm.put(e1.getKey(), na);
		}
		return nm;
	}

	@Deprecated
	public static Map<double[], Map<double[], Double>> getInverseDistanceMatrix(Collection<double[]> samples,
			Dist<double[]> gDist, double pow) {
		return getInverseDistanceMatrix(samples, gDist, pow, Double.MAX_VALUE);
	}

	public static Map<double[], Map<double[], Double>> getInverseDistanceMatrix(Collection<double[]> samples,
			Dist<double[]> gDist, double pow, double radius) {
		Map<double[], Map<double[], Double>> r = new HashMap<double[], Map<double[], Double>>();

		double minDist = -1;
		for (double[] a : samples) {
			Map<double[], Double> m = new HashMap<double[], Double>();

			for (double[] b : samples) {
				if (a == b)
					continue;

				double dist = gDist.dist(a, b);
				if (dist > radius)
					continue;

				if (dist == 0) { // a and b have same location
					if (minDist < 0) { // only calc/show message once
						minDist = Double.POSITIVE_INFINITY;
						for (double[] aa : samples) {
							for (double[] bb : samples) {
								if (aa == bb)
									continue;
								double d = gDist.dist(aa, bb);
								if (d > 0 && d < minDist)
									minDist = d;
							}
						}
						log.warn("Points with no distance present. Setting dist to " + minDist);
					}
					dist = minDist;
				}
				m.put(b, 1.0 / Math.pow(dist, pow));
			}
			r.put(a, m);
		}
		return r;
	}

	public static <T> Map<T, Map<T, Double>> getDistanceMatrix(Collection<T> samples, Dist<T> gDist,
			boolean withIdentity) {
		Map<T, Map<T, Double>> r = new HashMap<T, Map<T, Double>>();
		for (T a : samples) {
			Map<T, Double> m = new HashMap<T, Double>();
			for (T b : samples) {
				if (a == b && !withIdentity)
					continue;
				m.put(b, gDist.dist(a, b));
			}
			r.put(a, m);
		}
		return r;
	}

	public static Map<double[], List<double[]>> getKNNs(final List<double[]> samples, final Dist<double[]> gDist, int k,
			boolean includeIdentity) {
		Map<double[], List<double[]>> r = new HashMap<double[], List<double[]>>();

		for (final double[] x : samples) {
			PriorityQueue<double[]> pq = new PriorityQueue<>(samples.size(), new Comparator<double[]>() {
				@Override
				public int compare(double[] o1, double[] o2) {
					return Double.compare(gDist.dist(x, o1), gDist.dist(x, o2));
				}
			});
			pq.addAll(samples);
			List<double[]> sub = new ArrayList<double[]>();
			if (!includeIdentity)
				pq.poll(); // drop first/identity
			while (sub.size() < k)
				sub.add(pq.poll());
			r.put(x, sub);
		}
		return r;
	}

	// morans ------------------------------>>>
	public static double[] getMoransIStatistics(Map<double[], Map<double[], Double>> dMap, List<double[]> samples,
			List<Double> values) {
		Map<double[], Double> m = new HashMap<double[], Double>();
		for (int i = 0; i < samples.size(); i++)
			m.put(samples.get(i), values.get(i));
		return getMoransIStatistics(dMap, m);
	}

	public static double getMoransI(Map<double[], Map<double[], Double>> dMap, Map<double[], Double> values) {
		double n = values.size();
		double mean = 0;
		for (double[] d : values.keySet())
			mean += values.get(d) / n;

		// first term denominator
		double ftd = 0;
		for (double[] d : values.keySet())
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

	public static double[] getMoransIStatistics(Map<double[], Map<double[], Double>> dMap,
			Map<double[], Double> values) {
		double n = values.size();
		double moran = getMoransI(dMap, values);

		double E_I = -1.0 / (n - 1);

		// calculate variance, from wikipedia
		double s1 = 0;
		for (double[] i : dMap.keySet())
			for (double[] j : dMap.keySet())
				if (i != j) {
					double s = 0;
					if (dMap.get(i).containsKey(j))
						s += dMap.get(i).get(j);
					if (dMap.get(j).containsKey(i))
						s += dMap.get(j).get(i);
					s1 += Math.pow(s, 2);
				}

		s1 *= 0.5;

		double s2 = 0;
		for (double[] i : dMap.keySet()) {
			double s = 0;
			for (double[] j : dMap.keySet())
				if (i != j && dMap.get(i).containsKey(j))
					s += dMap.get(i).get(j);
			for (double[] j : dMap.keySet())
				if (i != j && dMap.get(j).containsKey(i))
					s += dMap.get(j).get(i);
			s2 += Math.pow(s, 2);
		}

		double m = 0;
		for (double d : values.values())
			m += d;
		m /= values.size();

		double s3 = 0;
		for (double d : values.values())
			s3 += Math.pow(d - m, 4);
		s3 /= n;

		double nom = 0;
		for (double d : values.values())
			nom += Math.pow(d - m, 2);
		s3 /= Math.pow(nom / n, 2);

		double sumWij = 0;
		for (double[] i : dMap.keySet())
			for (double[] j : dMap.keySet())
				if (i != j && dMap.get(i).containsKey(j))
					sumWij += dMap.get(i).get(j);

		double s4 = (Math.pow(n, 2) - 3 * n + 3) * s1 - s2 * n + 3 * Math.pow(sumWij, 2);
		double s5 = (Math.pow(n, 2) - n) * s1 - 2 * n * s2 + 6 * Math.pow(sumWij, 2);

		double Var_I = ((n * s4 - s3 * s5) / ((n - 1) * (n - 2) * (n - 3) * Math.pow(sumWij, 2))) - Math.pow(E_I, 2);
		double zScore = (moran - E_I) / Math.sqrt(Var_I);
		NormalDistribution nd = new NormalDistribution();

		return new double[] { moran, E_I, Var_I, zScore, 2 * nd.density(-Math.abs(zScore)), // TODO
																							// correct?
																							// shouln't
																							// we
																							// use
																							// cum
																							// dens?
		};
	}

	public static double[] getMoransIStatisticsMonteCarlo(Map<double[], Map<double[], Double>> dMap,
			Map<double[], Double> values, int reps) {
		double moran = getMoransI(dMap, values);

		DescriptiveStatistics ds = new DescriptiveStatistics();
		for (int i = 0; i < reps; i++) {

			// permute
			List<Double> l = new ArrayList<Double>(values.values());
			Collections.shuffle(l);
			Map<double[], Double> m = new HashMap<double[], Double>();
			int j = 0;
			for (double[] d : values.keySet())
				m.put(d, l.get(j++));

			ds.addValue(getMoransI(dMap, m));
		}

		double mean = ds.getMean();
		double var = ds.getStandardDeviation();
		double zScore = (moran - mean) / var;
		/*
		 * int i = 0; for( double permMoran : ds.getValues() ) { double
		 * permZScore = ( permMoran - mean ) / var; if( zScore > 0 && permZScore
		 * >= zScore ) i++; if( zScore < 0 && permZScore <= zScore ) i++; }
		 */

		// simplified calculation of i:
		int i = 0;
		for (double permMoran : ds.getValues()) {
			if (moran > mean && permMoran >= moran)
				i++;
			if (moran < mean && permMoran <= moran)
				i++;
		}

		return new double[] { moran, ds.getMean(), ds.getVariance(), zScore, // standard
																				// deviate
				// 2*new TDistribution(n-1).density(zScore), // p-Value???
				(double) i / reps, // p-Value!
		};
	}

	@Deprecated
	public static double getMoransI(Map<double[], Map<double[], Double>> dMap, int fa) {
		Set<double[]> samples = new HashSet<double[]>(dMap.keySet());
		for (double[] d : dMap.keySet())
			samples.addAll(dMap.get(d).keySet());

		Map<double[], Double> values = new HashMap<double[], Double>();
		for (double[] d : samples)
			values.put(d, d[fa]);
		return getMoransI(dMap, values);
	}

	private static double getIi(double[] s, Map<double[], Double> nbs, Map<double[], Double> values, double mean,
			double sd, double m2) {
		double ii = 0;
		for (Entry<double[], Double> nb : nbs.entrySet())
			ii += nb.getValue() * (values.get(nb.getKey()) - mean) / sd;
		return ii * (values.get(s) - mean) / m2;
	}

	public static List<double[]> getLocalMoransIMonteCarlo(List<double[]> samples, int idx,
			Map<double[], Map<double[], Double>> dMap, int reps) {
		Map<double[], Double> values = new HashMap<double[], Double>();
		for (double[] d : samples)
			values.put(d, d[idx]);
		return getLocalMoransIMonteCarlo(samples, values, dMap, reps);
	}

	// parameter samples necessary?
	public static List<double[]> getLocalMoransIMonteCarlo(List<double[]> samples, Map<double[], Double> values,
			Map<double[], Map<double[], Double>> dMap, int reps) {
		Random r = new Random();
		List<double[]> lisa = new ArrayList<double[]>();

		DescriptiveStatistics sampleDs = new DescriptiveStatistics();
		for (double d : values.values())
			sampleDs.addValue(d);
		double sampleMean = sampleDs.getMean();
		double sampleSd = 1.0;// sampleDs.getStandardDeviation();

		double m2 = 0;
		for (double d : values.values())
			m2 += Math.pow(d - sampleMean, 2);
		m2 /= values.size();

		for (double[] d : samples) {
			double ii = getIi(d, dMap.get(d), values, sampleMean, sampleSd, m2);

			DescriptiveStatistics iiDs = new DescriptiveStatistics();
			for (int i = 0; i < reps; i++) {

				// get random observation
				/*
				 * double[] rndD = null; while( rndD == null || rndD == d ) rndD
				 * = samples.get(r.nextInt(samples.size())); Map<double[],
				 * Double> nbs = dMap.get(rndD);
				 */

				Map<double[], Double> nbs = new HashMap<double[], Double>();
				for (Entry<double[], Double> e : dMap.get(d).entrySet())
					nbs.put(samples.get(r.nextInt(samples.size())), e.getValue());

				iiDs.addValue(getIi(d, nbs, values, sampleMean, sampleSd, m2));
			}
			double meanIi = iiDs.getMean();
			double sdIi = iiDs.getStandardDeviation();

			double zScore = (ii - meanIi) / sdIi;

			int i = 0;
			for (double permIi : iiDs.getValues()) {
				double permIiZScore = (permIi - meanIi) / sdIi;
				// two-tailed
				/*
				 * if( Math.abs(permIiZScore) >= Math.abs(zScore) ) i++;
				 */

				// one-tailed
				if (zScore >= 0 && permIiZScore >= zScore)
					i++;
				if (zScore < 0 && permIiZScore <= zScore)
					i++;
			}

			lisa.add(new double[] { ii, // lisa
					meanIi, // mean
					sdIi, zScore, (double) i / reps // p-Value
			});
		}
		return lisa;
	}

	// ------------------

	public static <T> void writeDistMatrixKeyValue(Map<T, Map<T, Double>> dMap, List<T> samples, File fn) {
		Map<T, Integer> idxMap = new HashMap<T, Integer>();
		for (int i = 0; i < samples.size(); i++)
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

	public static <T> Map<T, Map<T, Double>> readDistMatrixKeyValue(List<T> samples, File fn)
			throws NumberFormatException, IOException, FileNotFoundException {
		Map<T, Map<T, Double>> distMatrix = new HashMap<T, Map<T, Double>>();
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(fn));
			String line = br.readLine(); // ignore first line by reading but not
											// using
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
				if (br != null)
					br.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return distMatrix;
	}

	public static void main(String[] args) {
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("data/ozone.csv"), new int[] { 2, 3 },
				new int[] {}, true);
		List<double[]> samples = sdf.samples;
		Dist<double[]> gDist = new EuclideanDist(new int[] { 2, 3 });

		Map<double[], Map<double[], Double>> m1 = getInverseDistanceMatrix(samples, gDist, 1);
		Map<double[], Double> values = new HashMap<double[], Double>();
		for (double[] d : samples)
			values.put(d, d[1]);
		log.debug("Inv, 1, norm: " + getMoransI(getRowNormedMatrix(m1), values));
		log.debug(Arrays.toString(getMoransIStatistics(m1, values)));
		log.debug(Arrays.toString(getMoransIStatisticsMonteCarlo(m1, values, 100000)));
	}

	public static Map<double[], Map<double[], Double>> contiguityMapToDistanceMap(
			Map<double[], Set<double[]>> connectMap) {
		Map<double[], Map<double[], Double>> r = new HashMap<>();
		for (double[] a : connectMap.keySet()) {
			r.put(a, new HashMap<double[], Double>());
			for (double[] nb : connectMap.get(a))
				r.get(a).put(nb, 1.0);
		}
		return r;
	}

	public static Map<double[], Set<double[]>> getContiguityMap(List<double[]> samples, List<Geometry> geoms,
			boolean rookAdjacency, boolean includeIdentity) {
		Map<double[], Set<double[]>> r = new HashMap<>();
		for (int i = 0; i < samples.size(); i++) {
			Geometry a = geoms.get(i);
			Set<double[]> l = new HashSet<>();
			for (int j = 0; j < samples.size(); j++) {
				Geometry b = geoms.get(j);
				if (!includeIdentity && a == b)
					continue;
				if (!rookAdjacency) { // queen
					if (a.touches(b) || a.intersects(b))
						l.add(samples.get(j));
				} else { // rook
					if (a.intersection(b).getCoordinates().length > 1) // SLOW
						l.add(samples.get(j));
				}
			}
			r.put(samples.get(i), l);
		}
		return r;
	}

	public static boolean isContiugous(Map<double[], Set<double[]>> cm, Set<double[]> cluster) {
		if (cluster.isEmpty())
			return true;

		Set<double[]> visited = RegionUtils.getContiugousSubcluster(cm, cluster, cluster.iterator().next());
		if (visited.size() != cluster.size())
			return false;
		else
			return true;
	}
}
