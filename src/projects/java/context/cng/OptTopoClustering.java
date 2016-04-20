package context.cng;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;

import edu.uci.ics.jung.algorithms.cluster.EdgeBetweennessClusterer;
import edu.uci.ics.jung.graph.Graph;
import edu.uci.ics.jung.graph.UndirectedSparseGraph;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.Connection;
import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.utils.DataUtils;

// Topology representing network
public class OptTopoClustering {

	private static Logger log = Logger.getLogger(OptTopoClustering.class);

	public static void main(String args[]) {
		final Random r = new Random();

		int threads = 1;
		final int T_MAX = 100000;

		// 10/3 -> nmi 0.933
		final List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("output/rregions.shp"), new int[] {}, true);
		final List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(new File("output/rregions.shp"));
		
		final Map<Integer, Set<double[]>> cl = new HashMap<Integer, Set<double[]>>();
		for (double[] d : samples) {
			int c = (int) d[3];

			if (!cl.containsKey(c))
				cl.put(c, new HashSet<double[]>());

			cl.get(c).add(d);
		}

		int[] fa = new int[] { 2 };
		int[] ga = new int[] { 0, 1 };

		final Dist eDist = new EuclideanDist();
		final Dist gDist = new EuclideanDist(ga);
		final Dist fDist = new EuclideanDist(fa);

		ExecutorService es = Executors.newFixedThreadPool(threads);
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

		for (int n = 20; n < 120; n += 5) {
			for (int cns = 1; cns < n / 4; cns++) {
				for (int tInit = 0; tInit <= 400; tInit += 10) {
					for (int tFinal = tInit; tFinal <= 4000; tFinal += 100) {

						final int N = n;
						final int CNS = cns;
						final int TI = tInit;
						final int TF = tFinal;

						futures.add(es.submit(new Callable<double[]>() {

							@Override
							public double[] call() throws Exception {
								Map<Connection, Integer> cons = new HashMap<Connection, Integer>();
								Sorter bmuGetter = new KangasSorter(gDist, fDist, CNS);

								NG ng = new NG(N, (double) N / 2, 0.01, 0.5, 0.005, samples.get(0).length, bmuGetter);
								for (int t = 0; t < T_MAX; t++) {
									double[] x = samples.get(r.nextInt(samples.size()));
									ng.train((double) t / T_MAX, x);

									bmuGetter.sort(x, ng.getNeurons());
									cons.put(new Connection( ng.getNeurons().get(0), ng.getNeurons().get(1)), 0);

									// increase age of all cons
									for (Connection c : cons.keySet())
										cons.put(c, cons.get(c) + 1);

									double tA = TI * Math.pow(TF / TI, (double) t / T_MAX);

									List<Connection> drop = new ArrayList<Connection>();
									for (Connection c : cons.keySet())
										if (cons.get(c) > tA)
											drop.add(c);

									cons.keySet().removeAll(drop);
								}

								Map<double[], Set<double[]>> cluster = new HashMap<double[], Set<double[]>>();
								for (double[] w : ng.getNeurons())
									cluster.put(w, new HashSet<double[]>());
								for (double[] d : samples) {
									bmuGetter.sort(d, ng.getNeurons());
									double[] bmu = ng.getNeurons().get(0);
									cluster.get(bmu).add(d);
								}

								double nmiA = DataUtils.getNormalizedMutualInformation(cluster.values(), cl.values());

								log.debug("cons: "+cons.size());
								
								Graph<double[], String> g = new UndirectedSparseGraph<double[], String>();
								for (Connection c : cons.keySet()) {
									if (!g.getVertices().contains(c.getA()))
										g.addVertex(c.getA());
									if (!g.getVertices().contains(c.getB()))
										g.addVertex(c.getB());
									g.addEdge(g.getEdges().size() + "", c.getA(), c.getB());
								}

								// get used or connected ones
								Set<double[]> usedNeurons = new HashSet<double[]>();
								for (double[] d : cluster.keySet())
									if (!cluster.get(d).isEmpty() && !g.getVertices().contains(d))
										usedNeurons.add(d);

								for (int toRm = 1; toRm < 10; toRm++) {
									
									EdgeBetweennessClusterer<double[], String> clusterer = new EdgeBetweennessClusterer<double[], String>(toRm);
									Set<Set<double[]>> comm = clusterer.transform(g);

									// build cluster
									Set<Set<double[]>> fCluster = new HashSet<Set<double[]>>();
									for (Set<double[]> s : comm) {
										Set<double[]> ns = new HashSet<double[]>();
										for (double[] n : s)
											ns.addAll(cluster.get(n));
										fCluster.add(ns);
									}

									double nmiB = DataUtils.getNormalizedMutualInformation(fCluster, cl.values());
									
									log.debug(N + ":" + CNS + ":" + TI + ":" + TF + ":" + toRm + ":" + nmiA + ":" + nmiB);
								}
								return new double[]{};
							}
						}));

					}
				}
			}
		}
		es.shutdown();
	}
}
