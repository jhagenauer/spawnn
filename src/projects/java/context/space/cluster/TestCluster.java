package context.space.cluster;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
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
import spawnn.ng.ContextNG;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.sorter.SorterWMC;
import spawnn.ng.utils.NGUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;

public class TestCluster {
	private static Logger log = Logger.getLogger(TestCluster.class);

	public static void main(String args[]) {
		final Random r = new Random();

		final List<double[]> samples = new ArrayList<double[]>();
		final Map<Integer, Set<double[]>> classes = new HashMap<Integer, Set<double[]>>();
		for (double x = 0; x <= 3; x += 0.005) {
			int c;
			double v;
			if (x < 1.25) {
				c = 0;
				v = 0;
			} else if (x < 2.25) {
				c = 1;
				v = r.nextBoolean() ? 1 : 0;
			} else {
				c = 2;
				v = 1;
			}
			double[] d = new double[] { x, v };
			samples.add(d);

			if (!classes.containsKey(c))
				classes.put(c, new HashSet<double[]>());
			classes.get(c).add(d);
		}
		log.debug("samples: " + samples.size());

		DataUtils.writeCSV("output/test.csv", samples, new String[] { "x", "v" });

		final int[] fa = new int[] { 1 };
		final int[] ga = new int[] { 0 };

		final Dist<double[]> fDist = new EuclideanDist(fa);
		final Dist<double[]> gDist = new EuclideanDist(ga);

		// 0.72, 0.68
		final Map<double[], List<double[]>> knns = GeoUtils.getKNNs(samples, gDist, 10, true); 
		for( double[] a : knns.keySet() ) 
			knns.get(a).remove(a); 
		//final Map<double[],Map<double[],Double>> dMap = GeoUtils.knnsToWeights(knns);
		
		// 0.78
		final Map<double[], Map<double[], Double>> dMap = GeoUtils.getRowNormedMatrix(GeoUtils.getKNearestFromMatrix(GeoUtils.getInverseDistanceMatrix(samples, gDist, 2), 10));

		final int numCluster = classes.size();

		// DataUtils.normalizeColumns(samples, fa );
		// DataUtils.normalizeGeoColumns(samples, ga );

		final int T_MAX = 150000;
		int runs = 4;
		int threads = 4;

		// cng
		{
			log.debug("cng");
			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (int k = 1; k <= numCluster; k++) {
				for (int i = 0; i < runs; i++) {

					final int K = k;

					futures.add(es.submit(new Callable<double[]>() {
						@Override
						public double[] call() {
							Sorter<double[]> bmuGetter = new KangasSorter<double[]>(gDist, fDist, K);
							NG ng = new NG(numCluster, numCluster / 2, 0.01, 0.5, 0.005, samples.get(0).length, bmuGetter);

							for (int t = 0; t < T_MAX; t++) {
								double[] x = samples.get(r.nextInt(samples.size()));
								ng.train((double) t / T_MAX, x);
							}

							Map<double[], Set<double[]>> cluster = NGUtils.getBmuMapping(samples, ng.getNeurons(), bmuGetter);
							double nmi = DataUtils.getNormalizedMutualInformation(cluster.values(), classes.values());
							return new double[] { K, nmi };
						}
					}));
				}
			}
			es.shutdown();

			Map<Integer,DescriptiveStatistics> m = new HashMap<Integer,DescriptiveStatistics>();
			for (Future<double[]> f : futures) {
				try {
					double[] d = f.get();
					int k = (int)d[0];
					double nmi = d[1];
					if( !m.containsKey(k) )
						m.put(k,new DescriptiveStatistics());
					m.get(k).addValue(nmi);
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
			for( int k : m.keySet() )
				log.debug(k+": "+m.get(k).getMean());
		}
		
		// wdmng
		{
			log.debug("wdmng");
			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

			for (double alpha = 0.0; alpha <= 1; alpha += 0.02) {
				for (double beta = 0.0; beta <= 1; beta += 0.02) {
					for (int i = 0; i < runs; i++) {

						final double ALPHA = alpha, BETA = beta;

						futures.add(es.submit(new Callable<double[]>() {
							@Override
							public double[] call() {

								List<double[]> neurons = new ArrayList<double[]>();
								for (int i = 0; i < numCluster; i++) {
									double[] rs = samples.get(r.nextInt(samples.size()));
									double[] d = Arrays.copyOf(rs, rs.length * 2);
									for (int j = rs.length; j < d.length; j++)
										d[j] = r.nextDouble();
									neurons.add(d);
								}

								Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
								for (double[] d : samples)
									bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

								SorterWMC bg = new SorterWMC(bmuHist, dMap, fDist, ALPHA, BETA);
								ContextNG ng = new ContextNG(neurons, (double) neurons.size() / 2, 0.01, 0.5, 0.005, bg);

								bg.bmuHistMutable = true;
								for (int t = 0; t < T_MAX; t++) {
									double[] x = samples.get(r.nextInt(samples.size()));
									ng.train((double) t / T_MAX, x);

								}
								bg.bmuHistMutable = false;

								Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, neurons, bg);

								double nmi = DataUtils.getNormalizedMutualInformation(bmus.values(), classes.values());
								return new double[] { ALPHA, BETA, nmi };
							}
						}));
					}
				}
			}
			es.shutdown();
			
			Map<String,DescriptiveStatistics> m = new HashMap<String,DescriptiveStatistics>();
			for (Future<double[]> f : futures) {
				try {
					double[] d = f.get();
					double alpha = d[0];
					double beta = d[1];
					double nmi = d[2];
					
					String s = alpha+","+beta;
					if( !m.containsKey(s) )
						m.put(s,new DescriptiveStatistics());
					m.get(s).addValue(nmi);
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
			
			String bestS = null;
			double bestNMI = 0;
			for( String s : m.keySet() ) {
				if( bestS == null || bestNMI < m.get(s).getMean() ) {
					bestS = s;
					bestNMI = m.get(s).getMean();
				}	
			}
			log.debug("best: "+bestS+": "+bestNMI);
		}
	}
}
