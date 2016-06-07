package regionalization.medoid;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import spawnn.dist.ConstantDist;
import spawnn.dist.Dist;
import spawnn.utils.DataUtils;
import spawnn.utils.GraphUtils;

public class MedoidRegioClustering {

	public static enum GrowMode {
		WSS, EuclideanSqrd, WSS_INC
	};

	public static Map<double[], Set<double[]>> cluster(Map<double[], Set<double[]>> tree, Set<double[]> meds, Dist<double[]> dist, GrowMode dm, int maxNoImpro) {
		return cluster(tree, meds, dist, dm, maxNoImpro, dist, false);
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
		
	public static Map<double[], Set<double[]>> cluster(Map<double[], Set<double[]>> tree, Set<double[]> meds, Dist<double[]> dist, GrowMode dm, int maxNoImpro, Dist<double[]> updateDist, boolean nbSearch ) {
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
					if( nbSearch && !tree.get(m).contains(a) )
						continue;
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
	
	public static Map<double[], Set<double[]>> cluster2(Map<double[], Set<double[]>> tree, Set<double[]> meds, Dist<double[]> dist, GrowMode dm ) {	
		Set<double[]> medoids = new HashSet<double[]>(meds);
		
		double curCost = DataUtils.getWithinSumOfSquares(growFromMedoids(tree, medoids, dist, dm).values(), dist);
		while( true ) {			
			double bestInc = Double.NEGATIVE_INFINITY;
			double[] sMedoid = null, sElem = null;
			
			for( double[] m : new ArrayList<>(medoids) ) {
				medoids.remove(m);
				for( double[] nb : tree.get(m) ) {
					medoids.add(nb);	
					double inc = curCost - DataUtils.getWithinSumOfSquares(growFromMedoids(tree, medoids, dist, dm).values(), dist);
					if( inc > bestInc ) {
						bestInc = inc;
						sMedoid = m;
						sElem = nb;
					}
					medoids.remove(nb);
				}
				medoids.add(m);
			}
			
			if( bestInc < 0 )
				return growFromMedoids(tree, medoids, dist, dm);
									
			// do the swap
			medoids.remove(sMedoid);
			medoids.add(sElem);	
			curCost -= bestInc;
		}
	}
	
	public static Map<double[], Set<double[]>> cluster3(Map<double[], Set<double[]>> tree, Set<double[]> meds, Dist<double[]> dist, GrowMode dm, int maxNoImpro ) {
		Random ra = new Random();
		Set<double[]> medoids = new HashSet<double[]>(meds);
		Map<double[],Set<double[]>> cluster = growFromMedoids(tree, medoids, dist, dm);
		double cost = DataUtils.getWithinSumOfSquares(cluster.values(), dist);	
		
		int noImpro = 0;
		while( true ) {	
			
			// local search for random neighbor
			double[] m = new ArrayList<>(medoids).get(ra.nextInt(medoids.size()));
			double[] bestNB = null;
			double localCost =  DataUtils.getQuantizationError(m, cluster.get(m), dist);
			for( double[] nb : tree.get(m) ) {
				double c = DataUtils.getQuantizationError(nb, cluster.get(m), dist);
				if( c < localCost ) {
					localCost = c;
					bestNB = nb;
				} 
			}				
			
			if( bestNB != null ) {
				medoids.remove(m);
				medoids.add(bestNB);
				Map<double[],Set<double[]>> nc = growFromMedoids(tree, medoids, dist, dm);
				double c = DataUtils.getWithinSumOfSquares(nc.values(), dist);	
				System.out.println(c+" < "+cost +" ?" +(c<cost));
				if( c < cost ) {
					cost = c;
					cluster = nc;
					noImpro = 0;
				} else {
					medoids.remove(bestNB);
					medoids.add(m);
				}
			}
			if( noImpro++ > maxNoImpro )
				return growFromMedoids(tree, medoids, dist, dm);
		}
	}
	
	public enum MedoidInitMode {
		rnd, fDist, gDist, graphDist
	}
	
	public static Set<double[]> getInitMedoids( MedoidInitMode initMode, Map<double[],Set<double[]>> cm, Dist<double[]> fDist, Dist<double[]> gDist, int numCluster ) {
		Random ra = new Random();
		Set<double[]> medoids = new HashSet<double[]>();
		if( initMode == MedoidInitMode.rnd ) { // rnd init
			while( medoids.size() < numCluster ) {
				for (double[] s : cm.keySet())
					if (ra.nextDouble() < 1.0 / cm.keySet().size() ) {
						medoids.add(s);
						break;
					}
			}
		} else if( initMode == MedoidInitMode.fDist || initMode == MedoidInitMode.gDist || initMode == MedoidInitMode.graphDist ) { //k-means++-init
			List<double[]> samples = new ArrayList<>(GraphUtils.getNodes(cm));
			medoids.add(samples.get(ra.nextInt(samples.size())));
			Map<double[],Map<double[],Double>> wCm = GraphUtils.toWeightedGraph(cm,new ConstantDist<>(1.0));
			
			Map<double[],Map<double[],Double>> map = new HashMap<>();
			
			while( medoids.size() < numCluster ) {
				// build dist map
				Map<double[],Double> distMap = new HashMap<>();
				for( double[] n : samples) {
					if( medoids.contains(n) )
						continue;
					
					double d = Double.MAX_VALUE;
					if( initMode == MedoidInitMode.graphDist ) {
						for( double[] m : medoids ) {
							if( !map.containsKey(m) )
								map.put(m, GraphUtils.getShortestDists(wCm, m));
							d = Math.min(d, map.get(m).get(n) );
						}
					} else {
						for( double[] m : medoids )
							if( initMode == MedoidInitMode.fDist )
								d = Math.min( d, fDist.dist(n, m));
							else if( initMode == MedoidInitMode.gDist )
								d = Math.min( d, gDist.dist(n, m));
					}
					distMap.put( n, d );
				}
						
				// tournament selection
				double min = Collections.min(distMap.values());
				double max = Collections.max(distMap.values());
				double sum = 0;
				for( double d : distMap.values() )
					sum += Math.pow((d - min)/(max-min),2); 
					
				double v = ra.nextDouble() * sum;
				double lower = 0;
				for (Entry<double[], Double> e : distMap.entrySet()) {
					double w = Math.pow((e.getValue()-min)/(max-min),2);
					if (lower <= v && v <= lower + w ) {
						medoids.add(e.getKey());
						break;
					}
					lower += w;
				}		
				
			}
		} 
		return medoids;
	}
}
