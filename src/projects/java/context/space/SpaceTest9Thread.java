package context.space;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
import spawnn.ng.NG;
import spawnn.ng.ContextNG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.SorterWMC;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.RegionUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.Point;

public class SpaceTest9Thread {

	private static Logger log = Logger.getLogger(SpaceTest9Thread.class);

	public static void main(String[] args) {

		int threads = 15;
		int maxK = 100;

		final Random r = new Random();
		final int T_MAX = 150000;
		final int rcpFieldSize = 80;

		Map<String, double[]> results = new HashMap<String, double[]>();

		File file = new File("data/redcap/Election/election2004.shp");
		SpatialDataFrame sd = DataUtils.readShapedata(file, new int[] {}, true);
		final List<double[]> samples = sd.samples;
		final List<Geometry> geoms = sd.geoms;
		Map<double[], Set<double[]>> ctg = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");

		// build dist matrix and add coordinates to samples
		Map<double[], Map<double[], Double>> distMap = new HashMap<double[], Map<double[], Double>>();
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			Point p1 = geoms.get(i).getCentroid();

			distMap.put(d, new HashMap<double[], Double>());
			for (double[] nb : ctg.get(d)) {
				int j = samples.indexOf(nb);

				if (i == j)
					continue;

				Point p2 = geoms.get(j).getCentroid();
				distMap.get(d).put(nb, p1.distance(p2));
			}

			// ugly, but easy
			d[0] = p1.getX();
			d[1] = p1.getY();

			// normed coordinates
			d[2] = p1.getX();
			d[3] = p1.getY();
		}

		final int[] ga = new int[] { 0, 1 };
		final int[] gaNormed = new int[] { 2, 3 };

		final int fa = 7;
		final Dist<double[]> fDist = new EuclideanDist(new int[] { fa });
		final Dist<double[]> gDist = new EuclideanDist(ga);

		DataUtils.zScoreColumn(samples, fa);
		DataUtils.zScoreGeoColumns(samples, gaNormed, gDist);
		final Dist<double[]> normedGDist = new EuclideanDist(gaNormed);

		// build ctg-dist-Matrix
		Map<double[], Map<double[], Double>> cMap = new HashMap<double[], Map<double[], Double>>();
		for (double[] d : ctg.keySet()) {
			Map<double[], Double> dists = new HashMap<double[], Double>();
			for (double[] nb : ctg.get(d))
				if (d != nb)
					dists.put(nb, 1.0);

			cMap.put(d, dists);
		}

		// get knns
		final Map<double[], List<double[]>> knns = new HashMap<double[], List<double[]>>();
		for (double[] x : samples) {
			List<double[]> sub = new ArrayList<double[]>();
			while (sub.size() <= rcpFieldSize) { // sub.size() must be larger than cLength!

				double[] minD = null;
				for (double[] d : samples)
					if (!sub.contains(d) && (minD == null || gDist.dist(d, x) < gDist.dist(minD, x)))
						minD = d;
				sub.add(minD);
			}
			knns.put(x, sub);
		}
		log.debug("knn build.");

		for (final double w : new double[] { 0.0, 0.3 }) { 
			final String s = "WNG_" + w;
			log.info(s);

			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<double[]>> f = new ArrayList<Future<double[]>>();

			for (int k = 0; k < maxK; k++) {

				f.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {
						Map<Dist<double[]>, Double> map = new HashMap<Dist<double[]>, Double>();
						map.put(fDist, 1 - w);
						map.put(normedGDist, w);
						Dist<double[]> wDist = new WeightedDist<double[]>(map);

						DefaultSorter<double[]> bg = new DefaultSorter<double[]>(wDist);
						NG ng = new NG(9, 9.0 / 2, 0.01, 0.5, 0.005, samples.get(0).length, bg);

						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							ng.train((double) t / T_MAX, x);
						}

						Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
						return SpaceTest.getQuantizationError(samples, bmus, fDist, rcpFieldSize, knns);
					}
				}));
			}

			es.shutdown();

			double[] m = new double[rcpFieldSize];
			for (int dist = 0; dist < rcpFieldSize; dist++) {
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

		// geosom
		for (final int l : new int[] { 2 }) { // 2
			final String s = "GEOSOM_" + l;
			log.info(s);

			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<double[]>> f = new ArrayList<Future<double[]>>();

			for (int k = 0; k < maxK; k++) {

				f.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {

						spawnn.som.bmu.BmuGetter<double[]> bg = new spawnn.som.bmu.KangasBmuGetter<double[]>(normedGDist, fDist, l);
						Grid2D<double[]> grid = new Grid2DHex<double[]>(3, 3);
						SomUtils.initRandom(grid, samples);

						SOM som = new SOM(new GaussKernel(new LinearDecay(grid.getMaxDist(), 1)), new LinearDecay(1.0, 0.0), grid, bg);
						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							som.train((double) t / T_MAX, x);
						}

						Map<GridPos, Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bg);
						return SpaceTest.getQuantizationError(samples, bmus, fDist, rcpFieldSize, knns);
					}
				}));
			}
			es.shutdown();

			double[] m = new double[rcpFieldSize];
			for (int dist = 0; dist < rcpFieldSize; dist++) {
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
		for (final int l : new int[] { 6 }) { // 6
			final String s = "CNG_" + l;
			log.info(s);

			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<double[]>> f = new ArrayList<Future<double[]>>();

			for (int k = 0; k < maxK; k++) {

				f.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {

						KangasSorter<double[]> bg = new KangasSorter<double[]>(normedGDist, fDist, l);
						NG ng = new NG(9, 9.0 / 2, 0.01, 0.5, 0.005, samples.get(0).length, bg);

						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							ng.train((double) t / T_MAX, x);
						}

						Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
						return SpaceTest.getQuantizationError(samples, bmus, fDist, rcpFieldSize, knns);
					}
				}));
			}

			es.shutdown();

			double[] m = new double[rcpFieldSize];
			for (int dist = 0; dist < rcpFieldSize; dist++) {
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
		List<double[]> settings = new ArrayList<double[]>();
		settings.add(new double[] { -1, 0.85, 0.75 });

		for (double[] setting : settings) {
			final double alpha = setting[1], beta = setting[2], band = setting[0];
			final String s = "WMNG_" + alpha + "_" + beta + "_" + band;
			log.info(s);

			final Map<double[], Map<double[], Double>> dMap;
			if (band < 0) {
				dMap = new HashMap<double[], Map<double[], Double>>();
				for (double[] d : ctg.keySet()) {
					Map<double[], Double> dists = new HashMap<double[], Double>();
					for (double[] nb : ctg.get(d))
						dists.put(nb, 1.0);

					double n = dists.size();
					for (double[] nb : ctg.get(d))
						dists.put(nb, 1.0 / n);

					dMap.put(d, dists);
				}
			} else {
				dMap = SpaceTest.getDistMatrix(samples, gDist, band);
			}

			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<double[]>> f = new ArrayList<Future<double[]>>();

			for (int k = 0; k < maxK; k++) {

				f.add(es.submit(new Callable<double[]>() {

					@Override
					public double[] call() throws Exception {

						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < 9; i++) {
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
							ng.train((double) t / T_MAX, x);
						}
						bg.bmuHistMutable = false;

						Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
						return SpaceTest.getQuantizationError(samples, bmus, fDist, rcpFieldSize, knns);

					}
				}));

			}
			es.shutdown();

			double[] m = new double[rcpFieldSize];
			for (int dist = 0; dist < rcpFieldSize; dist++) {
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
			FileWriter fw = new FileWriter("output/election_results.csv");

			List<String> keys = new ArrayList<String>(results.keySet());

			StringBuffer header = new StringBuffer();
			for (String k : keys)
				header.append(k + ",");
			header.setCharAt(header.lastIndexOf(","), '\n');
			fw.write(header.toString());

			for (int dist = 0; dist < rcpFieldSize; dist++) {
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
}
