package regionalization.medoid;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import spawnn.dist.Dist;
import spawnn.utils.DataUtils;
import spawnn.utils.GraphUtils;

public class MedoidRegioClustering {

	// Actually, this is NOT PAM, but works nevertheless
	@Deprecated
	public static Map<double[], Set<double[]>> cluster(Map<double[],Set<double[]>> tree, int num, Dist<double[]> dist, boolean nbMode ) {
		Random r = new Random();

		Set<double[]> medoids = new HashSet<double[]>();
		while( medoids.size() < num ) {
			for (double[] s : tree.keySet())
				if (r.nextDouble() < 1.0 / tree.keySet().size() ) {
					medoids.add(s);
					break;
				}
		}
		
		Map<double[],Set<double[]>> bestCluster = null;
		double bestSum = 0;
		int noImpro = 0;
		
		while( true ) {
			// 1. init clusterMap with best medoids
			Map<double[], Map<double[], Set<double[]>>> clusterMap = new HashMap<double[], Map<double[], Set<double[]>>>();
			Map<double[],Set<double[]>> added = new HashMap<double[],Set<double[]>>();
			for( double[] m : medoids ) {
				clusterMap.put(m, new HashMap<double[],Set<double[]>>());
				clusterMap.get(m).put(m, new HashSet<double[]>() );
				added.put(m, new HashSet<double[]>());
				added.get(m).add(m);
			}
			
			// 2. Grow clusterMap
			Set<double[]> open = new HashSet<double[]>(GraphUtils.getNodes(tree));
			open.removeAll(clusterMap.keySet());
			while( !open.isEmpty() ) {
				
				double[] bestM = null, bestA = null, bestB = null;
				double bestCost = Double.MAX_VALUE;
				for( double[] m : clusterMap.keySet() ) {
					Set<double[]> nodes = GraphUtils.getNodes(clusterMap.get(m));
		
					for( double[] a : added.get(m) ) {
						for( double[] b : tree.get(a) ) 
							if( open.contains(b) ) { // unassigned
								nodes.add(b);
								double cost = DataUtils.getSumOfSquares(nodes, dist);
								if( bestM == null || cost < bestCost ) {
									bestM = m;
									bestA = a;
									bestB = b;
									bestCost = cost;
								}
								nodes.remove(b);
							}					
					}
				}
								
				if( !clusterMap.get(bestM).containsKey(bestA) )
					clusterMap.get(bestM).put(bestA, new HashSet<double[]>());
				clusterMap.get(bestM).get(bestA).add(bestB);
				if( !clusterMap.get(bestM).containsKey(bestB) )
					clusterMap.get(bestM).put(bestB, new HashSet<double[]>());
				clusterMap.get(bestM).get(bestB).add(bestA);
				added.get(bestM).add(bestB);
				open.remove(bestB);
			}
						
			Map<double[],Set<double[]>> clusters = new HashMap<double[],Set<double[]>>();
			for( Entry<double[],Map<double[],Set<double[]>>> e : clusterMap.entrySet() ) 
				clusters.put(e.getKey(),GraphUtils.getNodes(e.getValue()));
						
			double cost = DataUtils.getWithinSumOfSquares(clusters.values(), dist);
			if( bestCluster == null || cost < bestSum ) {
				//log.debug("found new best: "+cost+","+noImpro);
				bestSum = cost;
				bestCluster = clusters;
				noImpro = 0;
			}
						
			// 3. update medoids
			medoids.clear();
			for(double[] m : clusters.keySet() ) {
								
				double[] nm = m;
				double bs = Double.MAX_VALUE;				
				
				if( !nbMode ) {
					for( double[] a : clusters.get(m) ) {
						double sum = DataUtils.getSumOfSquares(a, clusters.get(m), dist);
						if( sum < bs ) {
							nm = a;
							bs = sum;
						}
					}
				} else {
					for( double[] a : clusterMap.get(m).get(m) ) {
						double sum = DataUtils.getSumOfSquares(a, clusters.get(m), dist);
						if( sum < bs ) {
							nm = a;
							bs = sum;
						}
					}
				}
					
				
				medoids.add(nm);
			}
						
			if( noImpro++ == 20 ) 
				break;
		}
		return bestCluster;
	}
	
	public static Map<double[], Set<double[]>> clusterCached(Map<double[],Set<double[]>> tree, int num , Dist<double[]> dist, DistMode dm ) {
		Random r = new Random();
		Set<double[]> medoids = new HashSet<double[]>();
		while( medoids.size() < num ) {
			for (double[] s : tree.keySet())
				if (r.nextDouble() < 1.0 / tree.keySet().size() ) {
					medoids.add(s);
					break;
				}
		}
		return clusterCached(tree, medoids, dist, dm );
	}
	
	public static enum DistMode { WSS, Euclidean, EuclideanSqrd };
	
	public static Map<double[], Set<double[]>> clusterCached(Map<double[],Set<double[]>> tree, Set<double[]> meds, Dist<double[]> dist, DistMode dm ) {
		
		Set<double[]> medoids = new HashSet<double[]>(meds);
		Map<double[],Set<double[]>> bestCluster = null;
		double bestSum = 0;
		
		for( int noImpro = 0; noImpro < 20; noImpro++ ) {
			// 1. init clusterMap with best medoids
			Map<double[], Map<double[], Set<double[]>>> clusterMap = new HashMap<double[], Map<double[], Set<double[]>>>();
			Map<double[],Set<double[]>> added = new HashMap<double[],Set<double[]>>();
			for( double[] m : medoids ) {
				clusterMap.put(m, new HashMap<double[],Set<double[]>>());
				clusterMap.get(m).put(m, new HashSet<double[]>() );
				added.put(m, new HashSet<double[]>());
				added.get(m).add(m);
			}
			
			// 2. Grow clusterMap
			class BestEntry {
				double[] bestA = null, bestB = null;
				double cost = Double.MAX_VALUE;
			}
			
			Map<double[],BestEntry> cache = new HashMap<double[],BestEntry>();
			
			Set<double[]> open = new HashSet<double[]>(GraphUtils.getNodes(tree));
			open.removeAll(clusterMap.keySet());
			while( !open.isEmpty() ) {
								
				double[] bestM = null;
				BestEntry beM = null;
				for( double[] m : clusterMap.keySet() ) {
					
					// get cost update cache
					if( !cache.containsKey(m) || !open.contains( cache.get(m).bestB) ) {
						BestEntry be = new BestEntry();
						Set<double[]> nodes = GraphUtils.getNodes(clusterMap.get(m));
						for( double[] a : added.get(m) ) {
							for( double[] b : tree.get(a) ) 
								if( open.contains(b) ) { // unassigned	
									
									if( dm == DistMode.WSS ) {
										nodes.add(b);
										double cost = DataUtils.getSumOfSquares(nodes, dist);
										if( cost < be.cost ) {
											be.bestA = a;
											be.bestB = b;
											be.cost = cost;
										}
										nodes.remove(b);
									} else if( dm == DistMode.Euclidean ) {
										double cost = dist.dist(m,b);
										if( cost < be.cost ) {
											be.bestA = a;
											be.bestB = b;
											be.cost = cost;
										}
									} else if( dm == DistMode.EuclideanSqrd ) {
										double cost = Math.pow(dist.dist(m,b),2);
										if( cost < be.cost ) {
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
					if( beM == null || be.cost < beM.cost ) {
						beM = be;
						bestM = m;
					}
				}
								
				if( !clusterMap.get(bestM).containsKey(beM.bestA) )
					clusterMap.get(bestM).put(beM.bestA, new HashSet<double[]>());
				clusterMap.get(bestM).get(beM.bestA).add(beM.bestB);
				if( !clusterMap.get(bestM).containsKey(beM.bestB) )
					clusterMap.get(bestM).put(beM.bestB, new HashSet<double[]>());
				clusterMap.get(bestM).get(beM.bestB).add(beM.bestA);
				
				added.get(bestM).add(beM.bestB);
				open.remove(beM.bestB);
				cache.remove(bestM);
			}
						
			Map<double[],Set<double[]>> clusters = new HashMap<double[],Set<double[]>>();
			for( Entry<double[],Map<double[],Set<double[]>>> e : clusterMap.entrySet() ) 
				clusters.put(e.getKey(),GraphUtils.getNodes(e.getValue()));
			
			double cost = DataUtils.getWithinSumOfSquares(clusters.values(), dist);
			if( bestCluster == null || cost < bestSum ) {
				//log.debug("found new best: "+cost+","+noImpro);
				bestSum = cost;
				bestCluster = clusters;
				noImpro = 0;
			}
						
			// 3. update medoids
			medoids = new HashSet<double[]>();
			for(double[] m : clusters.keySet() ) {
								
				double[] nm = m;
				double bs = Double.MAX_VALUE;				
				
				for( double[] a : clusters.get(m) ) {
					double sum = 0;
					if( dm == DistMode.WSS )
						sum = DataUtils.getSumOfSquares(a, clusters.get(m), dist);
					else if( dm == DistMode.Euclidean ) {
						for( double[] b : clusters.get(m) )
							sum += dist.dist(a, b);
					} else if( dm == DistMode.EuclideanSqrd )
						for( double[] b : clusters.get(m) )
							sum += Math.pow(dist.dist(a, b),2);
					if( sum < bs ) {
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
