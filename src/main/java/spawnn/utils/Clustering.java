package spawnn.utils;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;

public class Clustering {

	private static Logger log = Logger.getLogger(Clustering.class);

	// Actually, this is NOT PAM, but works nevertheless
	public static Map<double[], Set<double[]>> kMedoidsPAM(Collection<double[]> samples, int num, Dist<double[]> dist) {
		Random r = new Random();

		Map<double[], Set<double[]>> medoidMap = new HashMap<double[], Set<double[]>>();
		// 1. random init
		while (medoidMap.size() < num) {
			for (double[] s : samples)
				if (r.nextDouble() < 1.0 / samples.size() ) {
					medoidMap.put(s, new HashSet<double[]>());
					break;
				}
		}
		List<double[]> bestMedoids = null;
		double bestSum = 0;
		int noImpro = 0;
	
		while( true ) {			
			// 2. Assignment step
			for( double[] m : medoidMap.keySet() ) 
				medoidMap.get(m).add(m);
			
			for (double[] s : samples) {
				if( medoidMap.keySet().contains(s) )
					continue;
				
				double[] closest = null;
				for (double[] medoid : medoidMap.keySet()) 
					if (closest == null || dist.dist(s, medoid) < dist.dist(s, closest)) 
						closest = medoid;
				medoidMap.get(closest).add(s);
			}
			
			// 3. update step, get better medoid
			double sumCost = 0;
			List<double[]> newMedoids = new ArrayList<double[]>();
			for (Entry<double[],Set<double[]>> e : medoidMap.entrySet() ) {
				
				double bestCost = Double.MAX_VALUE;
				double[] bestMedoid = null;
				for( double[] d : e.getValue() ) {
					
					double cost = 0;
					for( double[] dd : e.getValue() )
						cost += dist.dist(dd, d);
					
					if( bestMedoid == null || cost < bestCost ) {
						bestMedoid = d;
						bestCost = cost;
					} 
				}
				newMedoids.add(bestMedoid);
				sumCost += bestCost;
			}
			
			if( bestMedoids == null || sumCost < bestSum ) {
				bestSum = sumCost;
				bestMedoids = newMedoids;
				noImpro = 0;
			}
			
			if( noImpro++ == 100 )
				break;
		
			medoidMap.clear();
			for( double[] m : newMedoids )
				medoidMap.put(m, new HashSet<double[]>() );	
		}
		return medoidMap;
	}
	
	public static Map<double[], Set<double[]>> kMeans(List<double[]> samples, int num, Dist<double[]> dist) {
		int length = samples.iterator().next().length;
		Random r = new Random();
		Map<double[], Set<double[]>> clusters = null;
		Set<double[]> centroids = new HashSet<double[]>();

		// get num unique(!) indices for centroids
		Set<Integer> indices = new HashSet<Integer>();
		while (indices.size() < num)
			indices.add(r.nextInt(samples.size()));

		for (int i : indices) {
			double[] d = samples.get(i);
			centroids.add(Arrays.copyOf(d, d.length));
		}

		int j = 0;
		boolean changed;
		do {
			clusters = new HashMap<double[], Set<double[]>>();
			for (double[] v : centroids)
				// init cluster
				clusters.put(v, new HashSet<double[]>());

			for (double[] s : samples) { // build cluster
				double[] nearest = null;
				double nearestCost = Double.MAX_VALUE;
				for (double[] c : clusters.keySet()) { // find nearest centroid
					double actCost = dist.dist(c, s);
					if (actCost < nearestCost) {
						nearestCost = actCost;
						nearest = c;
					}
				}
				clusters.get(nearest).add(s);
			}

			changed = false;
			for (double[] c : clusters.keySet()) {
				Collection<double[]> s = clusters.get(c);

				if (s.isEmpty())
					continue;

				// calculate new centroids
				double[] centroid = new double[length];
				for (double[] v : s)
					for (int i = 0; i < v.length; i++)
						centroid[i] += v[i];
				for (int i = 0; i < centroid.length; i++)
					centroid[i] /= s.size();

				// update centroids
				for (int i = 0; i < c.length; i++) {
					if (c[i] != centroid[i]) {
						centroids.remove(c);
						centroids.add(centroid);
						changed = true;
					}
				}
			}

			j++;
		} while (changed && j < 1000);

		return clusters;
	}

	public enum HierarchicalClusteringType {
		single_linkage, complete_linkage, average_linkage, ward
	};
	
	public static <T> List<Set<T>> cutTree( Map<Set<T>,TreeNode> tree, int numCluster ) {
		Comparator<TreeNode> comp = new Comparator<TreeNode>() {
			@Override
			public int compare(TreeNode o1, TreeNode o2) {
				return -Integer.compare(o1.age, o2.age);
			}
		};
		
		PriorityQueue<TreeNode> pq = new PriorityQueue<TreeNode>(1, comp);
		pq.add( Collections.min(tree.values(), comp));
		while( pq.size() < numCluster ) {
			for( TreeNode child : pq.poll().children )
				if( child != null )
					pq.add(child);
		}
		
		List<Set<T>> clusters = new ArrayList<Set<T>>();
		for( Entry<Set<T>,TreeNode> e : tree.entrySet() )
			if( pq.contains(e.getValue()))
				clusters.add(e.getKey());
		
		return clusters;
	}
	
	public static class TreeNode {
		public int age = 0;
		public double cost = 0; // sum of squares
		public TreeNode children[] = null;
		public String toString() { return age+", "+cost; }
	}
	
	// not connected
	public static Map<Set<double[]>,TreeNode> getHierarchicalClusterTree(List<double[]> samples, Dist<double[]> dist, HierarchicalClusteringType type) {
		return getHierarchicalClusterTree(samples, null, dist, type);
	}
	
	// connected
	public static Map<Set<double[]>,TreeNode> getHierarchicalClusterTree(Map<double[], Set<double[]>> cm, Dist<double[]> dist, HierarchicalClusteringType type) {
		return getHierarchicalClusterTree( null, cm, dist, type);
	}
	
	private static Map<Set<double[]>,TreeNode> getHierarchicalClusterTree( Collection<double[]> samples, Map<double[], Set<double[]>> cm, Dist<double[]> dist, HierarchicalClusteringType type) {
		if( samples == null ) {
			samples = new HashSet<double[]>();
			for( double[] a : cm.keySet() ) {
				samples.add(a);
				for( double[] b : cm.get(a) )
					samples.add(b);
			}
		}
		int length = samples.iterator().next().length;
		
		class FlatSet<T> extends HashSet<T> {
			private static final long serialVersionUID = -1960947872875758352L;
			public int hashCode = 0;
			
			@Override 
			public boolean add( T t ) {
				hashCode += t.hashCode();
				return super.add(t);
			}
			
			@Override
			public boolean addAll( Collection<? extends T> c ) {
				hashCode += c.hashCode();
				return super.addAll(c);
			}
			
			@Override
			public int hashCode() {
				return hashCode;
			}
		}
				
		Map<Set<double[]>,TreeNode> tree = new HashMap<Set<double[]>,TreeNode>();
		List<Set<double[]>> leafLayer = new ArrayList<Set<double[]>>();
		
		Map<Set<double[]>, Double> ssCache = new HashMap<Set<double[]>, Double>();
		Map<Set<double[]>, Map<Set<double[]>, Double>> unionCache = new HashMap<Set<double[]>, Map<Set<double[]>, Double>>();
						
		int age = 0;
		for (double[] d : samples) {
			Set<double[]> l = new FlatSet<double[]>();
			l.add(d);
			
			TreeNode cn = new TreeNode();
			cn.age = age;
			cn.cost = 0;
			tree.put(l,cn);
			
			leafLayer.add(l);
			ssCache.put(l, 0.0);
		}
		
				
		// init connected map
		Map<Set<double[]>, Set<Set<double[]>>> connected = null;
		if (cm != null) {
			connected = new HashMap<Set<double[]>, Set<Set<double[]>>>();
			for (Set<double[]> a : leafLayer) {
				for (Set<double[]> b : leafLayer) {
					if ( a != b && cm.get(a.iterator().next()).contains(b.iterator().next())) {
						if (!connected.containsKey(a))
							connected.put(a, new HashSet<Set<double[]>>());
						connected.get(a).add(b);
					}
				}
			}
		}
		
		while (leafLayer.size() > 1 ) {
			Set<double[]> c1 = null, c2 = null;
			double sMin = Double.MAX_VALUE;
			
			for (int i = 0; i < leafLayer.size() - 1; i++) {
				Set<double[]> l1 = leafLayer.get(i);

				for (int j = i + 1; j < leafLayer.size(); j++) {
					Set<double[]> l2 = leafLayer.get(j);
					
					if( connected != null 
							&& ( !connected.containsKey(l1) || !connected.get(l1).contains(l2) ) ) // disjoint
						continue;
															
					double s = -1;
					if (HierarchicalClusteringType.ward == type) {
																																				
						// get error sum of squares
						if (!unionCache.containsKey(l1) || !unionCache.get(l1).containsKey(l2)) {
																
							// calculate mean and ss, slightly faster than actually forming a union
							double[] r = new double[length];
							for (int l = 0; l < length; l++) {
								for (double[] d : l1)
									r[l] += d[l];
								for (double[] d : l2)
									r[l] += d[l];
							}
							
							for (int l = 0; l < length; l++)
								r[l] /= l1.size()+l2.size();
							
							double ssUnion = 0;
							for( double[] d : l1 ) {
								double di = dist.dist(r, d);
								ssUnion += di * di;
							}
							for( double[] d : l2 ) {
								double di = dist.dist(r, d);
								ssUnion += di * di;
							}

							if (!unionCache.containsKey(l1))
								unionCache.put( l1, new HashMap<Set<double[]>, Double>() );
							unionCache.get(l1).put(l2, ssUnion);
						}		
																		
						s = unionCache.get(l1).get(l2) - ( ssCache.get(l1) + ssCache.get(l2) );	
						
					} else if (HierarchicalClusteringType.single_linkage == type) {
						s = Double.MAX_VALUE;

						for (double[] d1 : l1) {
							for (double[] d2 : l2) {
								double d = dist.dist(d1, d2);
								if (d < s)
									s = d;
							}
						}
					} else if (HierarchicalClusteringType.complete_linkage == type) {
						s = Double.MIN_VALUE;

						for (double[] d1 : l1) {
							for (double[] d2 : l2) {
								double d = dist.dist(d1, d2);
								if (d > s)
									s = d;
							}
						}
					} else if (HierarchicalClusteringType.average_linkage == type) {
						s = 0;

						for (double[] d1 : l1) {
							for (double[] d2 : l2) {
								double d = dist.dist(d1, d2);
								s += d;
							}
						}
						s /= (l1.size() * l2.size());
					}

					if (s < sMin) {
						c1 = l1;
						c2 = l2;
						sMin = s;
					}
				}
			}
			
			boolean nanSS = false;
			if( c1 == null && c2 == null ) { // no connected clusters present anymore
				c1 = leafLayer.get(0);
				c2 = leafLayer.get(1);
				nanSS = true;	
			}
			
			// remove old clusters
			leafLayer.remove(c1);
			leafLayer.remove(c2);

			// merge
			Set<double[]> union = new FlatSet<double[]>();
			union.addAll(c1);
			union.addAll(c2);	
			leafLayer.add(union);
			
			double ss = 0;
			if( nanSS ) {
				ss = Double.NaN;
			} else if( type == HierarchicalClusteringType.ward ) {
				for( Set<double[]> s : leafLayer )
					if( s == union )
						ss += unionCache.get(c1).get(c2);
					else
						ss += ssCache.get(s);
				
				// update caches
				ssCache.remove(c1);
				ssCache.remove(c2);
				ssCache.put( union, unionCache.get(c1).get(c2) );
			
				unionCache.remove(c1);
			} else
				ss = DataUtils.getWithinSumOfSuqares(leafLayer, dist); // expensive
			
			// update connected map, non-connected cluster are ALSO merged in order to return a single tree. These merges have ss=NAN
			// 1. merge values of c1 and c2 and put union
			if( connected != null ) {
				Set<Set<double[]>> ns = connected.remove(c1);
				ns.addAll( connected.remove(c2) );
				connected.put(union, ns);
				
				// 2. replace all values c1,c2 by union
				for( Set<double[]> a : connected.keySet() ) {
					Set<Set<double[]>> s = connected.get(a);
					if( s.contains(c1) || s.contains(c2)) {
						s.remove(c1);
						s.remove(c2);
						s.add(union);
					}
				}
			}
						
			TreeNode cn = new TreeNode();
			cn.cost = ss;
			cn.age = ++age;
			cn.children = new TreeNode[]{ tree.get(c1), tree.get(c2) };
			tree.put(union, cn);
			
			//log.debug( leafLayer.size()+", ss:"+ss);
		}
		return tree;
	}
	
	public static List<Set<double[]>> skater(Map<double[], Set<double[]>> mst, int numCuts, Dist<double[]> dist, int minClusterSize) {

		class Edge { // undirected
			final double[] a, b;

			public Edge(double[] a, double[] b) {
				this.a = a;
				this.b = b;
			}

			public String toString() {
				return Arrays.toString(a) + Arrays.toString(b);
			};
		}

		int i = 0;
		while (i < numCuts) {

			Edge bestEdge = null;
			double bestEdgeValue = 0;
			int best_n = 0;

			// for each tree/subgraph
			for (Map<double[], Set<double[]>> sg : GraphUtils.getSubGraphs(mst)) {

				List<Edge> edges = new ArrayList<Edge>();
				for (double[] a : sg.keySet()) {
					for (double[] b : sg.get(a)) {
						if (a == b)
							continue;
						Edge e = new Edge(a, b);
						edges.add(e);
					}
				}
				Collections.shuffle(edges);

				if (edges.isEmpty()) {
					log.warn("Cannot split this subgraph any further. No edges.");
					continue;
				}

				// find init edge
				Edge initEdge = null;
				int diff = 0;
				// not all edges to safe comp. time
				for (Edge e : edges.subList(0, Math.min(edges.size(), 60))) {
					// remove edge
					sg.get(e.a).remove(e.b);
					sg.get(e.b).remove(e.a);

					Set<double[]> sgA = GraphUtils.getNodes(GraphUtils.getSubGraphOf(sg, e.a));
					Set<double[]> sgB = GraphUtils.getNodes(GraphUtils.getSubGraphOf(sg, e.b));
					int d = Math.abs(sgA.size() - sgB.size());
					if (initEdge == null || d < diff) {
						initEdge = e;
						diff = d;
					}

					// add edge
					sg.get(e.a).add(e.b);
					sg.get(e.b).add(e.a);
				}

				Map<Edge, Double> l = new HashMap<Edge, Double>();
				l.put(initEdge, 0.0);

				Map<Set<double[]>, Double> sgCache = new HashMap<Set<double[]>, Double>();

				for (int n = 0; n - best_n <= 60; n++) {

					if (l.isEmpty()) {
						log.warn("Cannot expand edges any further.");
						break;
					}

					// curEdge == best edge from l
					Edge curEdge = null;
					for (Edge e : l.keySet())
						if (curEdge == null || l.get(curEdge) < l.get(e) ) // 4354.172571482565
							curEdge = e;
					l.remove(curEdge);

					// expand neighbors
					Set<Edge> s_p = new HashSet<Edge>();
					for (double[] nb : sg.get(curEdge.a))
						if (nb != curEdge.b)
							s_p.add(new Edge(curEdge.a, nb));

					for (double[] nb : sg.get(curEdge.b))
						if (nb != curEdge.a)
							s_p.add(new Edge(curEdge.b, nb));

					// evaluate and store in L
					for (Edge e : s_p) {

						//double ssdT = DataUtils.getWithinClusterSumOfSuqares( getNodes( getSubGraphOf(sg, e.a) ), dist);
						double ssdT = -1;
						for (Set<double[]> s : sgCache.keySet())
							if (s.contains(e.a))
								ssdT = sgCache.get(s);

						if (ssdT == -1) {
							Set<double[]> s = GraphUtils.getNodes(GraphUtils.getSubGraphOf(sg, e.a));
							sgCache.put(s, DataUtils.getSumOfSquares(s, dist));
						}

						// remove edge
						sg.get(e.a).remove(e.b);
						sg.get(e.b).remove(e.a);

						Set<double[]> nodesA = GraphUtils.getNodes(GraphUtils.getSubGraphOf(sg, e.a));
						Set<double[]> nodesB = GraphUtils.getNodes(GraphUtils.getSubGraphOf(sg, e.b));

						double ssdT_a = DataUtils.getSumOfSquares(nodesA, dist);
						double ssdT_b = DataUtils.getSumOfSquares(nodesB, dist);

						double f1 = ssdT - ( ssdT_a + ssdT_b );
						if (Math.min(nodesA.size(), nodesB.size()) >= minClusterSize 
								&& (bestEdge == null || f1 > bestEdgeValue)) { // global best
							bestEdge = e;
							bestEdgeValue = f1;
							best_n = n;
						}

						double f2 = Math.min(ssdT - ssdT_a, ssdT - ssdT_b);
						l.put(e, f2);

						// add edge
						sg.get(e.a).add(e.b);
						sg.get(e.b).add(e.a);
					}
				}

			}

			// do the cut
			if (bestEdge != null) {
				mst.get(bestEdge.a).remove(bestEdge.b);
				mst.get(bestEdge.b).remove(bestEdge.a);
				i++;
			} else {
				log.warn("No cut possible, retrying..." + i);
			}
		}

		// form clusters
		List<Set<double[]>> clusters = new ArrayList<Set<double[]>>();
		for (Map<double[], Set<double[]>> sg : GraphUtils.getSubGraphs(mst))
			clusters.add(GraphUtils.getNodes(sg));

		return clusters;
	}
	
	public static void main(String[] args) {
		List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(new File("data/redcap/Election/election2004.shp"));
		List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/redcap/Election/election2004.shp"), new int[] {}, true);
		
		int[] fa = new int[] { 7 };

		for (int i : fa)
			DataUtils.zScoreColumn(samples, i);
		
		final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");
		final Dist<double[]> dist = new EuclideanDist(fa);
		
		int nrCluster = 7;
		Map<Set<double[]>,TreeNode> tree = Clustering.getHierarchicalClusterTree(samples, cm, dist, HierarchicalClusteringType.ward);
		System.out.println("WCSS1: " + DataUtils.getWithinSumOfSuqares(Clustering.cutTree( tree, nrCluster), dist));		
	}
}
