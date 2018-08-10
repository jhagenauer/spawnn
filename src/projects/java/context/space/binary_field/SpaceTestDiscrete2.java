package context.space.binary_field;

import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.ng.ContextNG;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.SorterWMC;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.Grid2D_Map;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class SpaceTestDiscrete2 {

	private static Logger log = Logger.getLogger(SpaceTestDiscrete2.class);

	public static void main(String[] args) {

		final List<double[]> samples = DataUtils.readCSV("data/somsd/grid100x100.csv");
		final Map<double[], Map<double[], Double>> dMap = SpaceTestDiscrete.readDistMap(samples, "data/somsd/grid100x100.wtg");

		/*
		 * List<double[]> samples =
		 * DataUtils.readCSV("data/somsd/grid1x10000.csv"); Map<double[],
		 * Map<double[], Double>> dMap =
		 * SpaceTestDiscrete.readDistMap(samples,"data/somsd/grid1x10000.wtg");
		 * 
		 * Map<double[],Set<double[]>> rmMap = new
		 * HashMap<double[],Set<double[]>>(); for( double[] x : dMap.keySet() )
		 * { for( double[] nb : dMap.get(x).keySet() ) if( nb[2] > x[2] ) { if(
		 * !rmMap.containsKey(x) ) rmMap.put(x, new HashSet<double[]>() );
		 * rmMap.get(x).add(nb); } } for( double[] x : rmMap.keySet() ) for(
		 * double[] nb : rmMap.get(x) ) dMap.get(x).remove(nb);
		 */

		// BinaryField2D.draw(new BinaryField2D(samples),
		// "output/dataGrid.png",false);
		// System.exit(1);

		final Random r = new Random();

		final boolean normed = true;
		int threads = 4;
		final int maxK = 1;
		final int T_MAX = 150000;
		final int maxDist = 5;
		final int maxRfSize = (maxDist + 1) * (maxDist + 2) * 2 - (maxDist + 1) * 4 + 1; // 2d,
		// rook
		// int maxRfSize = maxDist*2+1; // 1d, bidi
		// int maxRfSize = maxDist+1; // 1d, temp

		Map<String, double[]> results = new HashMap<String, double[]>();

		final int fa = 0;
		final int[] ga = new int[] { 1, 2 };
		final Dist<double[]> fDist = new EuclideanDist(new int[] { fa });
		final Dist<double[]> gDist = new EuclideanDist(ga);

		// wng
		for (final double w : new double[] { 0.07 }) { // 0.0, 0.07
			final String s = "WNG_" + w;
			log.info(s);

			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<double[]>> f = new ArrayList<Future<double[]>>();

			for (int k = 0; k < maxK; k++) {

				f.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
						Map<Dist<double[]>, Double> map = new HashMap<Dist<double[]>, Double>();
						map.put(gDist, w);
						map.put(fDist, 1 - w);
						Dist<double[]> wDist = new WeightedDist<double[]>(map);

						DefaultSorter<double[]> bg = new DefaultSorter<double[]>(wDist);

						NG ng = new NG(100, 50.0, 0.01, 0.5, 0.005, samples.get(0).length, bg);

						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							ng.train((double) t / T_MAX, x);
						}

						Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
						Map<double[], Set<Grid2D_Map<Boolean>>> rf = SpaceTestDiscrete.getReceptiveFields(samples, dMap, bmus, maxDist, maxRfSize, ga, fa);
						Map<double[], Grid2D_Map<Double>> mrfs = SpaceTestDiscrete2.getProbabilityFields(rf);

						// save most frequent DoubleFields
						double[] maxBmu = null;
						int max = 0;
						for( double[] bmu : rf.keySet() ) {
							if( maxBmu == null || rf.get(bmu).size() > max ) {
								maxBmu = bmu;
								max = rf.get(bmu).size();
							}
						}
												
						// reduce to 2
						DecimalFormat fo = new DecimalFormat("0000");
						Grid2D_Map<Double> df = mrfs.get(maxBmu);
						Grid2D_Map<Double> subDf = new Grid2D_Map<Double>(0,0);
						GridPos center = new GridPos(0,0);
						for (GridPos gp : df.getPositions()) 
							if(df.dist(gp, center) <= 2)
								subDf.setPrototypeAt(gp, df.getPrototypeAt(gp) );							
						String fn = "output/"+s+"_"+fo.format(max);
						
						for( int dist = 0; dist <= 2; dist++)
							log.debug(dist+"->"+getEntropy( subDf, dist, true) );
						
						return getUncertainty(rf, maxDist, normed);
					}

				}));
			}

			es.shutdown();

			double[] m = new double[maxDist + 1];
			for (int dist = 0; dist <= maxDist; dist++) {
				DescriptiveStatistics ds = new DescriptiveStatistics();
				for (Future<double[]> future : f) {
					try {
						ds.addValue(future.get()[dist]);
					} catch (InterruptedException e) {
						e.printStackTrace();
					} catch (ExecutionException e) {
						e.printStackTrace();
					}
				}
				m[dist] = ds.getMean();
			}
			log.debug(Arrays.toString(m));
			results.put(s, m);
		}
						
		// GeoSOM
		for (final int radius : new int[] { 3 }) { // 3
			final String s = "GEOSOM_" + radius;
			log.info(s);

			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<double[]>> f = new ArrayList<Future<double[]>>();

			for (int k = 0; k < maxK; k++) {

				f.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {

						spawnn.som.bmu.BmuGetter<double[]> bg = new spawnn.som.bmu.KangasBmuGetter<double[]>(gDist, fDist, radius);
						Grid2D<double[]> grid = new Grid2DHex<double[]>(10, 10);
						SomUtils.initRandom(grid, samples);

						SOM som = new SOM(new GaussKernel(new LinearDecay(10, 1)), new LinearDecay(1.0, 0.0), grid, bg);
						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							som.train((double) t / T_MAX, x);
						}

						Map<GridPos, Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bg);
						Map<GridPos, Set<Grid2D_Map<Boolean>>> rf = SpaceTestDiscrete.getReceptiveFields(samples, dMap, bmus, maxDist, maxRfSize, ga, fa);
						Map<GridPos, Grid2D_Map<Double>> mrfs = SpaceTestDiscrete2.getProbabilityFields(rf);
						
						// save most frequent DoubleFields
						GridPos maxBmu = null;
						int max = 0;
						for( GridPos bmu : rf.keySet() ) {
							if( maxBmu == null || rf.get(bmu).size() > max ) {
								maxBmu = bmu;
								max = rf.get(bmu).size();
							}
						}
												
						// reduce to 2
						DecimalFormat fo = new DecimalFormat("0000");
						Grid2D_Map<Double> df = mrfs.get(maxBmu);
						Grid2D_Map<Double> subDf = new Grid2D_Map<Double>(0,0);
						GridPos center = new GridPos(0,0);
						for (GridPos gp : df.getPositions()) 
							if(df.dist(gp, center) <= 2)
								subDf.setPrototypeAt(gp, df.getPrototypeAt(gp) );							
						String fn = "output/"+s+"_"+fo.format(max);
						
						for( int dist = 0; dist <= 2; dist++)
							log.debug(dist+"->"+getEntropy( subDf, dist, true) );
						
						return getUncertainty(rf, maxDist, normed);

					}
				}));
			}

			es.shutdown();

			double[] m = new double[maxDist + 1];
			for (int dist = 0; dist <= maxDist; dist++) {
				DescriptiveStatistics ds = new DescriptiveStatistics();
				for (Future<double[]> future : f) {
					try {
						ds.addValue(future.get()[dist]);
					} catch (InterruptedException e) {
						e.printStackTrace();
					} catch (ExecutionException e) {
						e.printStackTrace();
					}
				}
				m[dist] = ds.getMean();
			}
			log.debug(Arrays.toString(m));
			results.put(s, m);
		}

		// cng
		for (final int l : new int[] { 4 }) { // 4
			final String s = "CNG_" + l;
			log.info(s);

			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<double[]>> f = new ArrayList<Future<double[]>>();

			for (int k = 0; k < maxK; k++) {

				f.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {

						KangasSorter<double[]> bg = new KangasSorter<double[]>(gDist, fDist, l);
						NG ng = new NG(100, 50.0, 0.01, 0.5, 0.005, samples.get(0).length, bg);

						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							ng.train((double) t / T_MAX, x);
						}

						Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
						Map<double[], Set<Grid2D_Map<Boolean>>> rf = SpaceTestDiscrete.getReceptiveFields(samples, dMap, bmus, maxDist, maxRfSize, ga, fa);
						Map<double[], Grid2D_Map<Double>> mrfs = SpaceTestDiscrete2.getProbabilityFields(rf);
						
						// save most frequent DoubleFields
						double[] maxBmu = null;
						int max = 0;
						for( double[] bmu : rf.keySet() ) {
							if( maxBmu == null || rf.get(bmu).size() > max ) {
								maxBmu = bmu;
								max = rf.get(bmu).size();
							}
						}
												
						// reduce to 2
						DecimalFormat fo = new DecimalFormat("0000");
						Grid2D_Map<Double> df = mrfs.get(maxBmu);
						Grid2D_Map<Double> subDf = new Grid2D_Map<Double>(0,0);
						GridPos center = new GridPos(0,0);
						for (GridPos gp : df.getPositions()) 
							if(df.dist(gp, center) <= 2)
								subDf.setPrototypeAt(gp, df.getPrototypeAt(gp) );							
						String fn = "output/"+s+"_"+fo.format(max);
						
						for( int dist = 0; dist <= 2; dist++)
							log.debug(dist+"->"+getEntropy( subDf, dist, true) );

						return getUncertainty(rf, maxDist, normed);
					}
				}));
			}

			es.shutdown();

			double[] m = new double[maxDist + 1];
			for (int dist = 0; dist <= maxDist; dist++) {
				DescriptiveStatistics ds = new DescriptiveStatistics();
				for (Future<double[]> future : f) {
					try {
						ds.addValue(future.get()[dist]);
					} catch (InterruptedException e) {
						e.printStackTrace();
					} catch (ExecutionException e) {
						e.printStackTrace();
					}
				}
				m[dist] = ds.getMean();
			}
			log.debug(Arrays.toString(m));
			results.put(s, m);
		}

		// WMNG

		for (double[] setting : new double[][] { { 0.8, 0.2 } }) {
			final double alpha = setting[0], beta = setting[1];
			final String s = "WMNG_" + alpha + "_" + beta;
			log.info(s);

			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<double[]>> f = new ArrayList<Future<double[]>>();

			for (int k = 0; k < maxK; k++) {

				f.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {

						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < 100; i++) {
							double[] rs = samples.get(r.nextInt(samples.size()));
							double[] d = Arrays.copyOf(rs, rs.length * 2);
							for (int j = rs.length; j < d.length; j++)
								d[j] = r.nextDouble();
							neurons.add(d);
						}

						Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
						for (double[] d : samples)
							bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

						SorterWMC bg = new SorterWMC(bmuHist, dMap, fDist, alpha, beta);
						ContextNG ng = new ContextNG(neurons, (double) neurons.size() / 2, 0.01, 0.5, 0.005, bg);

						bg.bmuHistMutable = true;
						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							// double[] x = samples.get( t % samples.size() );
							ng.train((double) t / T_MAX, x);
						}
						bg.bmuHistMutable = false;

						Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, neurons, bg);
						Map<double[], Set<Grid2D_Map<Boolean>>> rf = SpaceTestDiscrete.getReceptiveFields(samples, dMap, bmus, maxDist, maxRfSize, ga, fa);
						Map<double[], Grid2D_Map<Double>> mrfs = SpaceTestDiscrete2.getProbabilityFields(rf);
						
						// save most frequent DoubleFields
						double[] maxBmu = null;
						int max = 0;
						for( double[] bmu : rf.keySet() ) {
							if( maxBmu == null || rf.get(bmu).size() > max ) {
								maxBmu = bmu;
								max = rf.get(bmu).size();
							}
						}
												
						// reduce to 2
						DecimalFormat fo = new DecimalFormat("0000");
						Grid2D_Map<Double> df = mrfs.get(maxBmu);
						Grid2D_Map<Double> subDf = new Grid2D_Map<Double>(0,0);
						GridPos center = new GridPos(0,0);
						for (GridPos gp : df.getPositions()) 
							if(df.dist(gp, center) <= 2)
								subDf.setPrototypeAt(gp, df.getPrototypeAt(gp) );							
						String fn = "output/"+s+"_"+fo.format(max);
						
						for( int dist = 0; dist <= 2; dist++)
							log.debug(dist+"->"+getEntropy( subDf, dist, true) );

						return getUncertainty(rf, maxDist, normed);

					}
				}));

			}
			es.shutdown();

			double[] m = new double[maxDist + 1];
			for (int dist = 0; dist <= maxDist; dist++) {
				DescriptiveStatistics ds = new DescriptiveStatistics();
				for (Future<double[]> future : f) {
					try {
						ds.addValue(future.get()[dist]);
					} catch (InterruptedException e) {
						e.printStackTrace();
					} catch (ExecutionException e) {
						e.printStackTrace();
					}
				}
				m[dist] = ds.getMean();
			}
			log.debug(Arrays.toString(m));
			results.put(s, m);
		}

		// write results as csv
		try {
			FileWriter fw = new FileWriter("output/raster_results.csv");

			List<String> keys = new ArrayList<String>(results.keySet());

			StringBuffer header = new StringBuffer();
			for (String k : keys)
				header.append(k + ",");
			header.setCharAt(header.lastIndexOf(","), '\n');
			fw.write(header.toString());

			for (int dist = 0; dist <= maxDist; dist++) {
				StringBuffer line = new StringBuffer();
				for (String k : keys)
					line.append(results.get(k)[dist] + ",");
				line.setCharAt(line.lastIndexOf(","), '\n');
				fw.write(line.toString());
			}

			fw.close();

		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	@Deprecated // ???
	public static Map<double[], Set<Map<Integer, int[]>>> getReceptiveFields(List<double[]> samples, final Map<double[], Map<double[], Double>> dMap, Map<double[], Set<double[]>> bmus, int maxDist, int rfMaxSize, int[] ga, int fa) {
		Map<double[], Set<Map<Integer, int[]>>> rcf = new HashMap<double[], Set<Map<Integer, int[]>>>();

		for (double[] bmu : bmus.keySet()) {
			Set<Map<Integer, int[]>> s = new HashSet<Map<Integer, int[]>>();

			for (double[] mp : bmus.get(bmu)) { // for each sample
				Map<Integer, int[]> m = new HashMap<Integer, int[]>();

				for (int dist = 0; dist <= maxDist; dist++) {
					Set<double[]> distSet = SpaceTestDiscrete.getSurrounding(mp, dMap, dist);
					if (dist > 0)
						distSet.removeAll(SpaceTestDiscrete.getSurrounding(mp, dMap, dist - 1));

					m.put(dist, new int[] { 0, 0 });
					for (double[] d : distSet)
						m.get(dist)[(int) d[fa]]++;
				}

				int c = 0;
				for (int[] i : m.values())
					c += i[0] + i[1];

				if (c == rfMaxSize)
					s.add(m);
			}
			if (!s.isEmpty())
				rcf.put(bmu, s);
		}
		return rcf;
	}

	@Deprecated // ???
	public static Map<double[], Map<Integer, int[]>> getIntersectReceptiveFields(Map<double[], Set<Map<Integer, int[]>>> rf) {
		Map<double[], Map<Integer, int[]>> r = new HashMap<double[], Map<Integer, int[]>>();

		for (double[] bmu : rf.keySet()) {
			Map<Integer, int[]> is = null;

			for (Map<Integer, int[]> m : rf.get(bmu)) {
				if (is == null)
					is = new HashMap<Integer, int[]>(m);

				// check
				for (int d : m.keySet())
					if (is.containsKey(d) && !Arrays.equals(is.get(d), m.get(d))) {
						for (int d2 : m.keySet())
							if (d2 >= d)
								is.remove(d2);
					}

			}
			r.put(bmu, is);

		}
		return r;
	}

	// hier gibts wohl ein problem
	public static <T> Map<T, Grid2D_Map<Double>> getProbabilityFields(Map<T, Set<Grid2D_Map<Boolean>>> rcp) {
		Map<T, Grid2D_Map<Double>> r = new HashMap<T, Grid2D_Map<Double>>();
		for (Entry<T,Set<Grid2D_Map<Boolean>>> bmu : rcp.entrySet() ) {
			
			Grid2D_Map<Double> df = new Grid2D_Map<Double>(0,0);
			for (Grid2D_Map<Boolean> bf : bmu.getValue() ) {
				for (GridPos gp : bf.getPositions()) {

					if (!df.getPositions().contains(gp))
						df.setPrototypeAt(gp, 0.0);

					if (bf.getPrototypeAt(gp))
						df.setPrototypeAt(gp, df.getPrototypeAt(gp) + 1);
				}
			}
			for (GridPos gp : df.getPositions()) {
				double d = df.getPrototypeAt(gp) / bmu.getValue().size(); // normalize
				df.setPrototypeAt(gp, d);
			}
			r.put(bmu.getKey(), df);
		}
		return r;
	}

	public static <T> double[] getUncertainty(Map<T, Set<Grid2D_Map<Boolean>>> rf, int maxDist, boolean rate) {
		double sum = 0;
		for (Set<Grid2D_Map<Boolean>> s : rf.values())
			sum += s.size();

		double[] r = new double[maxDist + 1];
		Map<T, Grid2D_Map<Double>> mrfs = SpaceTestDiscrete2.getProbabilityFields(rf);
				
		for (int dist = 0; dist <= maxDist; dist++) {

			for (T bmu : mrfs.keySet()) {
				double q = getEntropy(mrfs.get(bmu), dist, rate);
				double p_k = rf.get(bmu).size() / sum;
				r[dist] += p_k * q;
			}
		}
		return r;
	}
	
	public static double getEntropy(Grid2D_Map<Double> df, double dist, boolean rate) {
		GridPos center = new GridPos(0, 0);
		double q = 0;
		int num = 0;
		for (GridPos gp : df.getPositions()) {
			if (df.dist(gp, center) == dist) {
				double p = df.getPrototypeAt(gp);
				if (p != 0)
					q += p * Math.log(p) / Math.log(2);
				if (p != 1)
					q += (1 - p) * Math.log(1 - p) / Math.log(2);
				num++;
			} 
		}
		if (rate)
			q /= num;
		return - q;
	}
}
