package landCon;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.Drawer;
import spawnn.utils.RegionUtils;

public class LandCon {

	private static Logger log = Logger.getLogger(LandCon.class);

	private static List<TreeNode> getHierarchicalClusterTree(Collection<double[]> samples, Map<double[], Set<double[]>> cm, Dist<double[]> dist, HierarchicalClusteringType type, int threads) {
		int length = samples.iterator().next().length;

		class FlatSet<T> extends HashSet<T> {
			private static final long serialVersionUID = -1960947872875758352L;
			public int hashCode = 0;

			@Override
			public boolean add(T t) {
				hashCode += t.hashCode();
				return super.add(t);
			}

			@Override
			public boolean addAll(Collection<? extends T> c) {
				hashCode += c.hashCode();
				return super.addAll(c);
			}

			@Override
			public int hashCode() {
				return hashCode;
			}
		}

		List<TreeNode> tree = new ArrayList<>();
		Map<TreeNode, Set<double[]>> curLayer = new HashMap<>();

		Map<TreeNode, Double> ssCache = new HashMap<>();
		Map<TreeNode, Map<TreeNode, Double>> unionCache = new HashMap<>();

		int age = 0;
		for (double[] d : samples) {
			TreeNode cn = new TreeNode();
			cn.age = age;
			cn.cost = 0;
			cn.contents = new FlatSet<double[]>();
			cn.contents.add(d);
			tree.add(cn);

			Set<double[]> l = new FlatSet<double[]>(); // flat
			l.add(d);
			curLayer.put(cn, l);
			ssCache.put(cn, 0.0);
		}

		// init connected map
		final Map<TreeNode, Set<TreeNode>> connected = new HashMap<TreeNode, Set<TreeNode>>();
		for (TreeNode a : curLayer.keySet())
			for (TreeNode b : curLayer.keySet())
				if (a != b && cm.get(a.contents).contains(b.contents)) {
					if (!connected.containsKey(a))
						connected.put(a, new HashSet<TreeNode>());
					connected.get(a).add(b);
				}

		while (curLayer.size() > 1) {
			TreeNode c1 = null, c2 = null;
			double sMin = Double.MAX_VALUE;

			List<TreeNode> cl = new ArrayList<>(curLayer.keySet());

			ExecutorService es = Executors.newFixedThreadPool(threads);
			List<Future<double[]>> futures = new ArrayList<Future<double[]>>();
			for (int t = 0; t < threads; t++) {
				final int T = t;

				futures.add(es.submit(new Callable<double[]>() {
					@Override
					public double[] call() throws Exception {
						int c1 = -1, c2 = -1;
						double sMin = Double.MAX_VALUE;

						for (int i = T; i < cl.size() - 1; i += threads) {
							TreeNode l1 = cl.get(i);

							for (int j = i + 1; j < cl.size(); j++) {
								TreeNode l2 = cl.get(j);

								if (!connected.containsKey(l1) || !connected.get(l1).contains(l2)) // disjoint
									continue;

								double s = Double.NaN;
								if (HierarchicalClusteringType.ward == type) {
									// get error sum of squares
									if (!unionCache.containsKey(l1) || !unionCache.get(l1).containsKey(l2)) {

										// calculate mean and sum of squares,
										// slightly faster than actually forming
										// a union
										double[] mean = new double[length];
										for (int l = 0; l < length; l++) {
											for (double[] d : curLayer.get(l1))
												mean[l] += d[l];
											for (double[] d : curLayer.get(l2))
												mean[l] += d[l];
										}

										for (int l = 0; l < length; l++)
											mean[l] /= curLayer.get(l1).size() + curLayer.get(l2).size();

										double ssUnion = 0;
										for (double[] d : curLayer.get(l1)) {
											double di = dist.dist(mean, d);
											ssUnion += di * di;
										}
										for (double[] d : curLayer.get(l2)) {
											double di = dist.dist(mean, d);
											ssUnion += di * di;
										}

										// since no other thread uses l1 we do
										// not need synchronization
										if (!unionCache.containsKey(l1))
											unionCache.put(l1, new HashMap<TreeNode, Double>());
										unionCache.get(l1).put(l2, ssUnion);
									}
									s = unionCache.get(l1).get(l2) - (ssCache.get(l1) + ssCache.get(l2));
								} else if (HierarchicalClusteringType.single_linkage == type) {
									s = Double.MAX_VALUE;
									for (double[] d1 : curLayer.get(l1))
										for (double[] d2 : curLayer.get(l2))
											s = Math.min(s, dist.dist(d1, d2));
								} else if (HierarchicalClusteringType.complete_linkage == type) {
									s = Double.MIN_VALUE;
									for (double[] d1 : curLayer.get(l1))
										for (double[] d2 : curLayer.get(l2))
											s = Math.max(s, dist.dist(d1, d2));
								} else if (HierarchicalClusteringType.average_linkage == type) {
									s = 0;
									for (double[] d1 : curLayer.get(l1))
										for (double[] d2 : curLayer.get(l2))
											s += dist.dist(d1, d2);
									s /= (curLayer.get(l1).size() * curLayer.get(l2).size());
								}
								if (s < sMin) {
									c1 = i;
									c2 = j;
									sMin = s;
								}
							}
						}
						return new double[] { c1, c2, sMin };
					}
				}));
			}
			es.shutdown();
			try {
				for (Future<double[]> f : futures) {
					double[] d = f.get();
					if (d[2] < sMin) {
						c1 = cl.get((int) d[0]);
						c2 = cl.get((int) d[1]);
						sMin = d[2];
					}
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}

			if (c1 == null && c2 == null) { // no connected clusters present
											// anymore
				log.debug("only non-connected clusters present! " + curLayer.size());
				return tree;
			}

			// create merge node, remove c1,c2
			TreeNode mergeNode = new TreeNode();
			Set<double[]> union = new FlatSet<double[]>();
			union.addAll(curLayer.remove(c1));
			union.addAll(curLayer.remove(c2));
			mergeNode.age = ++age;
			mergeNode.children = Arrays.asList(new TreeNode[] { c1, c2 });

			// calculate/update cost
			double ss = 0;
			if (type == HierarchicalClusteringType.ward) {
				ss = unionCache.get(c1).get(c2);
				for (TreeNode s : curLayer.keySet())
					ss += ssCache.get(s);

				// update caches
				ssCache.remove(c1);
				ssCache.remove(c2);
				ssCache.put(mergeNode, unionCache.get(c1).get(c2));

				unionCache.remove(c1);
			} else
				ss = sMin;
			mergeNode.cost = ss;

			// add nodes
			curLayer.put(mergeNode, union);
			tree.add(mergeNode);

			// update connected map
			if (connected != null) {
				// 1. merge values of c1 and c2 and put union
				Set<TreeNode> ns = connected.remove(c1);
				ns.addAll(connected.remove(c2));
				connected.put(mergeNode, ns);

				// 2. replace all values c1,c2 by merged node
				for (Set<TreeNode> s : connected.values()) {
					if (s.contains(c1) || s.contains(c2)) {
						s.remove(c1);
						s.remove(c2);
						s.add(mergeNode);
					}
				}
			}

		}
		return tree;
	}
	
	public static int[][] getCluster(double[][] samples, double[][] cMat, int[] fa, String clusterType, int[] numCluster, int threads) {
		List<double[]> sa = new ArrayList<>();
		for (double[] d : samples)
			sa.add(Arrays.copyOf(d, d.length));

		Map<double[], Set<double[]>> cm = new HashMap<double[], Set<double[]>>();
		for (int i = 0; i < sa.size(); i++) {
			Set<double[]> s = new HashSet<>();
			for (int j = 0; j < cMat[i].length; j++)
				if (cMat[i][j] != 0)
					s.add(sa.get(j));
			cm.put(sa.get(i), s);
		}

		HierarchicalClusteringType type = null;
		if (clusterType.equals("W"))
			type = HierarchicalClusteringType.ward;
		else if (clusterType.equals("A"))
			type = HierarchicalClusteringType.average_linkage;
		else if (clusterType.equals("C"))
			type = HierarchicalClusteringType.complete_linkage;
		else if (clusterType.equals("S"))
			type = HierarchicalClusteringType.single_linkage;

		List<TreeNode> tree = getHierarchicalClusterTree(sa, cm, new EuclideanDist(fa), type, threads );

		int[][] r = new int[numCluster.length][sa.size()];
		for (int i = 0; i < numCluster.length; i++) {
			List<Set<double[]>> cluster = Clustering.cutTree(tree, numCluster[i]);
			for (int j = 0; j < sa.size(); j++) {
				for (int k = 0; k < cluster.size(); k++) {
					if (cluster.get(k).contains(sa.get(j)))
						r[i][j] = k;
				}
			}
		}
		return r;
	}

	public static void main(String[] args) {
		List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(new File("data/redcap/Election/election2004.shp"));
		List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/redcap/Election/election2004.shp"), new int[] {}, true);
		int[] fa = new int[] { 7 };
		DataUtils.transform(samples, fa, Transform.zScore);
		final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");
		final Dist<double[]> dist = new EuclideanDist(fa);

		/*
		 * SpatialDataFrame sdf =
		 * DataUtils.readSpatialDataFrameFromShapefile(new
		 * File("R:/data/gemeinden.shp"), true); List<Geometry> geoms =
		 * sdf.geoms.subList(0, 4000); List<double[]> samples =
		 * sdf.samples.subList(0, 4000); int[] fa = new int[] { 4, 5, 6, 7, 8, 9
		 * }; Set<double[]> rmSamples = new HashSet<>(); Set<Geometry> rmGeoms =
		 * new HashSet<>(); for (int i = 0; i < samples.size(); i++) { double[]
		 * d = samples.get(i); for (int j : fa) if (Double.isNaN(d[j])) {
		 * rmSamples.add(d); rmGeoms.add(geoms.get(i)); } }
		 * samples.removeAll(rmSamples); geoms.removeAll(rmGeoms);
		 * DataUtils.transform(samples, fa, Transform.zScore); Map<double[],
		 * Set<double[]>> cm = GraphUtils.deriveQueenContiguitiyMap(samples,
		 * geoms, false); GraphUtils.writeContiguityMap(cm, samples,
		 * "output/gemeinden.ctg"); final Dist<double[]> dist = new
		 * EuclideanDist(fa);
		 */
		log.debug(samples.size());

		int nrCluster = 7;
		{ // old
			long time = System.currentTimeMillis();
			List<TreeNode> tree = Clustering.getHierarchicalClusterTree(cm, dist, HierarchicalClusteringType.ward);
			List<Set<double[]>> ct = Clustering.cutTree(tree, nrCluster);
			log.debug("Nr cluster: " + ct.size());
			log.debug("Within sum of squares: " + DataUtils.getWithinSumOfSquares(ct, dist) + ", took: " + (System.currentTimeMillis() - time) / 1000.0);
			Drawer.geoDrawCluster(ct, samples, geoms, "output/clustering.png", true);
		}

		{ // threaded
			long time = System.currentTimeMillis();
			List<TreeNode> tree = getHierarchicalClusterTree(samples, cm, dist, HierarchicalClusteringType.ward, 4);
			List<Set<double[]>> ct = Clustering.cutTree(tree, nrCluster);
			log.debug("Nr cluster: " + ct.size());
			log.debug("Within sum of squares: " + DataUtils.getWithinSumOfSquares(ct, dist) + ", took: " + (System.currentTimeMillis() - time) / 1000.0);
			Drawer.geoDrawCluster(ct, samples, geoms, "output/clustering2.png", true);
		}
	}
}
