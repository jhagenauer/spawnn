package lisa;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
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

import com.vividsolutions.jts.geom.Geometry;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.ContextNG;
import spawnn.ng.sorter.SorterWMC;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.ClusterValidation;
import spawnn.utils.ColorBrewer;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class WMNGLisa {

	private static Logger log = Logger.getLogger(WMNGLisa.class);

	public static void main(String[] args) {

		final int T_MAX = 1000000;
		final Random r = new Random();
		final int reps = 10000;
		int threads = 4;

		File file = new File("data/redcap/Election/election2004.shp");
		final SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(file, true);
		final List<double[]> samples = sdf.samples;
		final List<Geometry> geoms = sdf.geoms;
		final Map<double[], Map<double[], Double>> dMap = GeoUtils.getRowNormedMatrix(GeoUtils.listsToWeights(GeoUtils.getContiguityMap(samples, geoms, false, false)));

		final int fa = 7; // bush pct
		DataUtils.zScoreColumn(samples, fa);
		final Dist<double[]> fDist = new EuclideanDist(new int[] { fa });
		// ------------------------------------------------------------------------

		Map<double[],Double> v = new HashMap<double[],Double>();
		for( double[] d : samples )
			v.put(d, d[fa]);
		List<double[]> lisa = GeoUtils.getLocalMoransIMonteCarlo( samples, v, dMap, reps);
		Drawer.geoDrawValues(geoms, lisa, 0, sdf.crs, ColorBrewer.Blues, "output/lisa_mc.png");

		List<Double> values = new ArrayList<Double>();
		for (double[] d : lisa)
			if (d[4] < 0.0001)
				values.add(0.0);
			else if (d[4] < 0.001)
				values.add(1.0);
			else if (d[4] < 0.01)
				values.add(2.0);
			else if (d[4] < 0.05)
				values.add(3.0);
			else
				values.add(4.0);
		Drawer.geoDrawValues(geoms, values, sdf.crs, ColorBrewer.Spectral, "output/lisa_mc_signf.png");

		DescriptiveStatistics ds = new DescriptiveStatistics();
		for (double[] d : samples)
			ds.addValue(d[fa]);
		double mean = ds.getMean();

		final Map<Integer, Set<double[]>> lisaCluster = new HashMap<Integer, Set<double[]>>();
		for (int i = 0; i < samples.size(); i++) {
			double[] l = lisa.get(i);
			double[] d = samples.get(i);
			int clust = -1;

			if (l[4] > 0.05) // not significant
				clust = 0;
			else if (l[0] > 0 && d[fa] > mean)
				clust = 1; // high-high
			else if (l[0] > 0 && d[fa] < mean)
				clust = 2; // low-low
			else if (l[0] < 0 && d[fa] > mean)
				clust = 3; // high-low
			else if (l[0] < 0 && d[fa] < mean)
				clust = 4; // low-high
			else
				clust = 5; // unknown

			if (!lisaCluster.containsKey(clust))
				lisaCluster.put(clust, new HashSet<double[]>());
			lisaCluster.get(clust).add(d);
		}
		Drawer.geoDrawCluster(lisaCluster.values(), samples, geoms, "output/lisa_mc_clust.png", false);

		for (Entry<Integer, Set<double[]>> e : lisaCluster.entrySet())
			log.debug(e.getKey() + ":" + e.getValue().size());

		System.exit(1);
		// -----------------------------------------------------------------------------------------

		String fn = "output/lisa_wmng.csv";
		try {
			String s = "nrNeurons,alpha,beta,nmi\n";
			Files.write(Paths.get(fn), s.getBytes());
		} catch (IOException e) {
			e.printStackTrace();
		}

		List<double[]> params = new ArrayList<double[]>();
		for (int nrNeurons : new int[] { 5 })
		for( double nbInit : new double[] {nrNeurons*2.0/3} )
		for( double nbFinal : new double[] {0.1} )
		for( double lrInit : new double[] {0.6} )
		for( double lrFinal : new double[] {0.01} )
		for (double alpha : new double[]{0.5,0.55, 0.6, 0.65})
		for (double beta : new double[] { 0.0 })
		params.add(new double[] { nrNeurons, nbInit, nbFinal, lrInit, lrFinal, alpha, beta });

		ExecutorService es = Executors.newFixedThreadPool(threads);
		Map<double[], Future<double[]>> futures = new HashMap<double[], Future<double[]>>();

		for (final double[] p : params) {
			futures.put(p, es.submit(new Callable<double[]>() {

				@Override
				public double[] call() {
					
					int nrNeurons = (int) p[0];
					double nbInit = p[1];
					double nbFinal = p[2];
					double lrInit = p[3];
					double lrFinal = p[4];
					double alpha = p[5];
					double beta = p[6];

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

					SorterWMC sorter = new SorterWMC(bmuHist, dMap, fDist, alpha, beta);
					DecayFunction nbRate = new PowerDecay( nbInit, nbFinal );
					DecayFunction adaptRate = new PowerDecay( lrInit, lrFinal );
					ContextNG ng = new ContextNG(neurons, nbRate, adaptRate, sorter);

					sorter.bmuHistMutable = true;
					for (int t = 0; t < T_MAX; t++) {
						double[] x = samples.get(r.nextInt(samples.size()));
						ng.train((double) t / T_MAX, x);
					}
					sorter.bmuHistMutable = false;

					List<double[]> stats = new ArrayList<double[]>();
					for (double[] d : samples) {
						sorter.sort(d, neurons);
						double[] bmu = neurons.get(0);

						double dist = sorter.getDist(d, bmu);

						DescriptiveStatistics distDs = new DescriptiveStatistics();
						for (int i = 0; i < reps; i++) {

							List<double[]> nbs = new ArrayList<double[]>(dMap.get(d).keySet());

							// shuffle hist of neighbors [alternative: change context vector?]
							List<double[]> nbHist = new ArrayList<double[]>();
							for (int j = 0; j < nbs.size(); j++) {
								double[] nb = nbs.get(j);
								nbHist.add(bmuHist.get(nb));
								bmuHist.put(nb, neurons.get(r.nextInt(neurons.size())));
							}

							distDs.addValue(sorter.getDist(d, bmu));

							// restore
							for (int j = 0; j < nbs.size(); j++)
								bmuHist.put(nbs.get(j), nbHist.get(j));
						}
						double meanDist = distDs.getMean();
						double sdDist = distDs.getStandardDeviation();

						double zScore = (dist - meanDist) / sdDist;

						int i = 0;
						for (double permDist : distDs.getValues()) {
							double permDistZScore = (permDist - meanDist) / sdDist;

							// one-tailed
							if (zScore >= 0 && permDistZScore >= zScore)
								i++;
							if (zScore < 0 && permDistZScore <= zScore)
								i++;
						}

						stats.add(new double[] { dist, // dist
								meanDist, // mean
								sdDist, zScore, (double) i / reps // p-Value
						});
					}

					List<double[]> fixOrderNeurons = new ArrayList<double[]>(neurons);
					Map<Integer, Set<double[]>> signfCluster = new HashMap<Integer, Set<double[]>>();
					for (int i = 0; i < samples.size(); i++) {
						double[] d = samples.get(i);

						int idx = 0;
						if (stats.get(i)[4] <= 0.05) {
							sorter.sort(d, neurons);
							idx = fixOrderNeurons.indexOf(neurons.get(0)) + 1;
						}
						if (!signfCluster.containsKey(idx))
							signfCluster.put(idx, new HashSet<double[]>());
						signfCluster.get(idx).add(d);
					}
					double nmi = ClusterValidation.getNormalizedMutualInformation(signfCluster.values(), lisaCluster.values());
					Drawer.geoDrawCluster(signfCluster.values(), samples, geoms, "output/wmng_mc_clust_" + Arrays.toString(p) + ".png", false);
					log.debug(Arrays.toString(p)+","+nmi);
					return new double[] { nmi };
				}
			}));
		}
		es.shutdown();

		try {
			for (Entry<double[], Future<double[]>> e : futures.entrySet()) {
				String s = "";
				for (double d : e.getKey())
					s += d + ",";
				s += e.getValue().get()[0] + "\n";
				Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
			}
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InterruptedException e1) {
			e1.printStackTrace();
		} catch (ExecutionException e1) {
			e1.printStackTrace();
		}

	}
}
