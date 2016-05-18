package regionalization.medoid;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import spawnn.dist.Dist;
import spawnn.utils.DataUtils;
import spawnn.utils.GraphUtils;

public class MedoidRegioClustering {

	public static enum GrowMode {
		WSS, EuclideanSqrd, WSS_INC
	};

	public static Map<double[], Set<double[]>> cluster(Map<double[], Set<double[]>> tree, Set<double[]> meds, Dist<double[]> dist, GrowMode dm, int maxNoImpro) {
		return cluster(tree, meds, dist, dm, maxNoImpro, dist);
	}

	public static Map<double[], Set<double[]>> growFromMedoids(Map<double[], Set<double[]>> tree, Set<double[]> medoids, Dist<double[]> dist, GrowMode dm) {
		// 1. init clusterMap with best medoids
		Map<double[], Map<double[], Set<double[]>>> clusterMap = new HashMap<double[], Map<double[], Set<double[]>>>();
		Map<double[], Set<double[]>> added = new HashMap<double[], Set<double[]>>();
		for (double[] m : medoids) {
			clusterMap.put(m, new HashMap<double[], Set<double[]>>());
			clusterMap.get(m).put(m, new HashSet<double[]>());

			added.put(m, new HashSet<double[]>());
			added.get(m).add(m);
		}

		// 2. Grow clusterMap
		class BestEntry {
			double[] bestA = null, bestB = null;
			double cost = Double.MAX_VALUE;
		}

		Map<double[], BestEntry> cache = new HashMap<double[], BestEntry>();

		Set<double[]> open = new HashSet<double[]>(GraphUtils.getNodes(tree));
		open.removeAll(clusterMap.keySet());
		while (!open.isEmpty()) {

			double[] bestM = null;
			BestEntry beM = null;
			for (double[] m : clusterMap.keySet()) {

				// get cost update cache
				if (!cache.containsKey(m) || !open.contains(cache.get(m).bestB)) {
					BestEntry be = new BestEntry();
					Set<double[]> nodes = added.get(m);
					double preCost = 0;

					if (dm == GrowMode.WSS_INC)
						preCost = DataUtils.getSumOfSquares(nodes, dist);
					for (double[] a : new ArrayList<double[]>(nodes)) {
						for (double[] b : tree.get(a)) {
							if (!open.contains(b))
								continue; 
							if (dm == GrowMode.WSS_INC) {
								nodes.add(b);
								double cost = DataUtils.getSumOfSquares(nodes, dist) - preCost;
								if (cost < be.cost) {
									be.bestA = a;
									be.bestB = b;
									be.cost = cost;
								}
								nodes.remove(b);
							} else if (dm == GrowMode.WSS) {
								nodes.add(b);
								double cost = DataUtils.getSumOfSquares(nodes, dist);
								if (cost < be.cost) {
									be.bestA = a;
									be.bestB = b;
									be.cost = cost;
								}
								nodes.remove(b);
							} else if (dm == GrowMode.EuclideanSqrd) {
								double cost = Math.pow(dist.dist(m, b), 2);
								if (cost < be.cost) {
									be.bestA = a;
									be.bestB = b;
									be.cost = cost;
								}
							}
						}
					}
					cache.put(m, be);
				}

				BestEntry be = cache.get(m);
				if (beM == null || be.cost < beM.cost) {
					beM = be;
					bestM = m;
				}
			}

			if (!clusterMap.get(bestM).containsKey(beM.bestA))
				clusterMap.get(bestM).put(beM.bestA, new HashSet<double[]>());
			clusterMap.get(bestM).get(beM.bestA).add(beM.bestB);
			if (!clusterMap.get(bestM).containsKey(beM.bestB))
				clusterMap.get(bestM).put(beM.bestB, new HashSet<double[]>());
			clusterMap.get(bestM).get(beM.bestB).add(beM.bestA);

			added.get(bestM).add(beM.bestB);
			open.remove(beM.bestB);
		}

		Map<double[], Set<double[]>> clusters = new HashMap<double[], Set<double[]>>();
		for (Entry<double[], Map<double[], Set<double[]>>> e : clusterMap.entrySet())
			clusters.put(e.getKey(), GraphUtils.getNodes(e.getValue()));

		return clusters;
	}
		
	public static Map<double[], Set<double[]>> cluster(Map<double[], Set<double[]>> tree, Set<double[]> meds, Dist<double[]> dist, GrowMode dm, int maxNoImpro, Dist<double[]> updateDist) {
		Map<double[], Set<double[]>> bestCluster = null;
		double bestCost = 0;
		
		int noImpro = 0;
		Set<double[]> medoids = new HashSet<double[]>(meds);
		while (true) {
			
			Map<double[], Set<double[]>> clusters = growFromMedoids(tree, medoids, dist, dm);
			double cost = DataUtils.getWithinSumOfSquares(clusters.values(), dist);
			if (bestCluster == null || cost < bestCost) {
				bestCost = cost;
				bestCluster = clusters;
				noImpro = 0;
			} 

			if (noImpro++ >= maxNoImpro)
				break;

			// 3. update medoids
			medoids.clear();
			for (double[] m : clusters.keySet()) {
				double[] nm = m;
				double bs = Double.MAX_VALUE;

				for (double[] a : clusters.get(m)) {
					double sum = DataUtils.getSumOfSquares(a, clusters.get(m), updateDist);
					if (sum < bs) {
						nm = a;
						bs = sum;
					}
				}
				medoids.add(nm);
			}
		}
		return bestCluster;
	}
}
