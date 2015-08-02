package wmng.ga2;

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

import org.apache.log4j.Logger;

import regionalization.RegionUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.ContextNG;
import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.sorter.SorterWMC;
import spawnn.ng.utils.NGUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.Point;

public class WMNGParamSensREDCAP_sameCluster {

	private static Logger log = Logger.getLogger(WMNGParamSensREDCAP_sameCluster.class);

	public static void main(String[] args) {

		final int T_MAX = 150000;
		final int runs = 64;
		final int threads = 4;
		final int nrNeurons = 10;
		final Random r = new Random();

		File file = new File("data/redcap/Election/election2004.shp");
		final List<double[]> samples = DataUtils.readSamplesFromShapeFile(file, new int[] {}, true);
		final List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(file);

		// build dist matrix and add coordinates to samples
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			Point p1 = geoms.get(i).getCentroid();

			// ugly, but easy
			d[0] = p1.getX();
			d[1] = p1.getY();

			// normed coordinates
			d[2] = p1.getX();
			d[3] = p1.getY();
		}

		final int[] ga = new int[] { 0, 1 };
		final int[] gaNormed = new int[] { 2, 3 };

		final int fa = 7; // bush pct
		final int fips = 4; // county_f basically identical to fips
		final Dist<double[]> fDist = new EuclideanDist(new int[] { fa });
		final Dist<double[]> gDist = new EuclideanDist(ga);

		DataUtils.zScoreColumn(samples, fa);
		DataUtils.zScoreGeoColumns(samples, gaNormed, gDist);

		final Map<double[], Set<double[]>> ctg = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");

		Map<double[], Map<double[], Double>> dCtgMap = new HashMap<double[], Map<double[], Double>>();
		for (double[] d : ctg.keySet()) {
			dCtgMap.put(d, new HashMap<double[], Double>());
			for (double[] nb : ctg.get(d))
				dCtgMap.get(d).put(nb, 1.0);
		}
		final Map<double[], Map<double[], Double>> dMap = GeoUtils.getRowNormedMatrix(dCtgMap);
		final double[] a = getSampleByFips(samples, fips, 48383);
		final double[] b = getSampleByFips(samples, fips, 48311);
		
		log.debug("fDist: "+fDist.dist(a, b));
		log.debug("gDist: "+gDist.dist(a,b));
		log.debug("ctgDist: "+fDist.dist(DataUtils.getMeanClusterElement(ctg.get(a)), DataUtils.getMeanClusterElement(ctg.get(b))));
		
		long time = System.currentTimeMillis();
		try {
			FileWriter fw = new FileWriter("output/cng_redcap_sameCluster.csv");
			fw.write("# runs " + runs +"\n");
			fw.write("l,dummy,sameCluster\n");

			List<double[]> params = new ArrayList<double[]>();
			for (int i = 1; i <= nrNeurons; i++)
				params.add(new double[] { i });

			for (final double[] param : params) {
				log.debug(Arrays.toString(param));

				ExecutorService es = Executors.newFixedThreadPool(threads);
				List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

				for (int run = 0; run < runs; run++) {
					futures.add(es.submit(new Callable<double[]>() {

						@Override
						public double[] call() {

							List<double[]> neurons = new ArrayList<double[]>();
							for (int i = 0; i < nrNeurons; i++) {
								double[] rs = samples.get(r.nextInt(samples.size()));
								neurons.add(Arrays.copyOf(rs, rs.length));
							}

							Sorter<double[]> bg = new KangasSorter<double[]>(gDist, fDist, (int) param[0]);
							NG ng = new NG(neurons, (double) neurons.size() / 2, 0.01, 0.5, 0.005, bg);

							for (int t = 0; t < T_MAX; t++) {
								double[] x = samples.get(r.nextInt(samples.size()));
								ng.train((double) t / T_MAX, x);
							}

							Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
							return new double[]{ sameCluster(bmus, new double[][]{a,	b,} ) ? 1.0 : 0.0, };
						}
					}));

				}
				es.shutdown();
				fw.write(param[0] + ",0,");

				for (int i = 0; i < futures.iterator().next().get().length; i++) {
					double mean = 0;
					for (Future<double[]> f : futures)
						mean += f.get()[i] / runs;
					fw.write("," + mean);
				}
				fw.write("\n");
			}
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
		log.debug("took: " + (System.currentTimeMillis() - time) / 1000 + "s");
		
		time = System.currentTimeMillis();
		try {
			FileWriter fw = new FileWriter("output/wmng_redcap_sameCluster.csv");
			fw.write("# runs " + runs +"\n");
			fw.write("alpha,beta,sameCluster\n");
			
			double step = 0.05;
			List<double[]> params = new ArrayList<double[]>();
			for (double alpha = 0.0; alpha <= 1; alpha += step, alpha = Math.round(alpha * 10000) / 10000.0)
				for (double beta = 0.0; beta <= 1; beta += step, beta = Math.round(beta * 10000) / 10000.0) {
					double[] d = new double[] { alpha, beta };
					params.add(d);
				}

			for (final double[] param : params) {
				log.debug(Arrays.toString(param));

				ExecutorService es = Executors.newFixedThreadPool(threads);
				List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

				for (int run = 0; run < runs; run++) {
					futures.add(es.submit(new Callable<double[]>() {

						@Override
						public double[] call() {

							List<double[]> neurons = new ArrayList<double[]>();
							for (int i = 0; i < nrNeurons; i++) {
								double[] rs = samples.get(r.nextInt(samples.size()));
								double[] d = Arrays.copyOf(rs, rs.length * 2);
								for (int j = rs.length; j < d.length; j++)
									d[j] = r.nextDouble();
								neurons.add(d);
							}

							Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
							for (double[] d : samples)
								bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

							SorterWMC bg = new SorterWMC(bmuHist, dMap, fDist, param[0], param[1]);
							ContextNG ng = new ContextNG(neurons, (double) neurons.size() / 2, 0.01, 0.5, 0.005, bg);

							bg.bmuHistMutable = true;
							for (int t = 0; t < T_MAX; t++) {
								double[] x = samples.get(r.nextInt(samples.size()));
								ng.train((double) t / T_MAX, x);
							}
							bg.bmuHistMutable = false;

							Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
							return new double[]{ sameCluster(bmus, new double[][]{a,	b,} ) ? 1.0 : 0.0, };
						}
					}));

				}
				es.shutdown();
				fw.write(param[0] + "," + param[1]);

				for (int i = 0; i < futures.iterator().next().get().length; i++) {
					double mean = 0;
					for (Future<double[]> f : futures)
						mean += f.get()[i] / runs;
					fw.write("," + mean);
				}
				fw.write("\n");
			}
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
		log.debug("took: " + (System.currentTimeMillis() - time) / 1000 + "s");
	}

	public static boolean sameCluster(Map<double[], Set<double[]>> bmus, double[][] samples) {
		for (Set<double[]> s : bmus.values()) {
			int count = 0;
			for (double[] a : s)
				for (double[] b : samples)
					if (a == b)
						count++;
			if (count == samples.length)
				return true;
		}
		return false;
	}

	public static double[] getSampleByFips(List<double[]> samples, int fc, int fips) {
		for (double[] d : samples)
			if (d[fc] == fips)
				return d;
		return null;
	}
}
