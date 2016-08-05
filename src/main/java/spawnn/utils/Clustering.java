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
import spawnn.utils.DataUtils.Transform;

public class Clustering {

	private static Logger log = Logger.getLogger(Clustering.class);

	// Actually, this is NOT PAM, but works nevertheless, needs some rework/cleanup
	public static Map<double[], Set<double[]>> kMedoidsPAM(Collection<double[]> samples, int num, Dist<double[]> dist) {
		Random r = new Random();
		Set<double[]> medoids = new HashSet<double[]>();
		while( medoids.size() < num ) {
			for (double[] s : samples)
				if (r.nextDouble() < 1.0 / samples.size() ) {
					medoids.add(s);
					break;
				}
		}
		
		Map<double[],Set<double[]>> bestCluster = null;
		double bestSum = 0;
		int noImpro = 0;
	
		while( true ) {	
			// 1. random init
			Map<double[], Set<double[]>> clusterMap = new HashMap<double[], Set<double[]>>();
			for( double[] m : medoids ) {
				clusterMap.put(m, new HashSet<double[]>());
				clusterMap.get(m).add(m);
			}
			
			// 2. Assignment step
			for( double[] m : clusterMap.keySet() ) 
				clusterMap.get(m).add(m);
			
			double sumCost = 0;
			for (double[] s : samples) {
				if( clusterMap.keySet().contains(s) )
					continue;
				
				double[] closest = null;
				for (double[] medoid : clusterMap.keySet()) 
					if (closest == null || dist.dist(s, medoid) < dist.dist(s, closest)) 
						closest = medoid;
				sumCost += dist.dist(closest, s);
				clusterMap.get(closest).add(s);
			}
			
			if( bestCluster == null || sumCost < bestSum ) {
				bestSum = sumCost;
				bestCluster = clusterMap;
				noImpro = 0;
			}
						
			// 3. update medoids
			medoids.clear();
			for (Entry<double[],Set<double[]>> e : clusterMap.entrySet() ) {
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
				medoids.add(bestMedoid);
			}
						
			if( noImpro++ == 200 )
				break;	
		}
		return bestCluster;
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
		
	public static List<Set<double[]>> cutTree( Collection <TreeNode> tree, int numCluster ) {
		Comparator<TreeNode> comp = new Comparator<TreeNode>() {
			@Override
			public int compare(TreeNode o1, TreeNode o2) {
				return -Integer.compare(o1.age, o2.age);
			}
		};
		
		// multiple for constrained clustering
		Set<TreeNode> roots = new HashSet<>(tree);
		for( TreeNode a : tree )
			for( TreeNode b : tree )
				if( b.children.contains(a) ) // a is children of b and therefore no root
						roots.remove(a);
		
		PriorityQueue<TreeNode> pq = new PriorityQueue<TreeNode>(1, comp);
		pq.addAll(roots);
		while( pq.size() < numCluster ) { 
			TreeNode tn = pq.poll();
			if( tn == null )
				throw new RuntimeException("To few observations for the desired number of clusters!");
			
			for( TreeNode child : tn.children )
				if( child != null )
					pq.add(child);
		}
		
		List<Set<double[]>> clusters = new ArrayList<Set<double[]>>();
		for( TreeNode t : pq )
			clusters.add( getContents(t) );		
		return clusters;
	}
	
	// Gets contents of leafs for node
	public static Set<double[]> getContents( TreeNode node ) {
		Set<double[]> contents = new HashSet<>();
		List<TreeNode> l = new ArrayList<>();
		l.add(node);
		
		while( !l.isEmpty() ) {
			TreeNode cur = l.remove(0);			
			for( TreeNode child : cur.children )
				if( child != null )
					l.add(child);
			
			if( cur.age == 0 ) // leaf-layer
				contents.addAll(cur.contents);
		}
		return contents;
	}
	
	public static class TreeNode {
		public Set<double[]> contents = null; // only nodes of age 0 should ever have contents! 
		public int age = 0;
		public double cost = 0; 
		public List<TreeNode> children = new ArrayList<TreeNode>();
		public String toString() { return "["+age+", "+cost+"]"; }
	}
		
	public static Map<double[],Set<double[]>> toREDCAPSpanningTree( Collection<TreeNode> tree, Map<double[],Set<double[]>> cm, HierarchicalClusteringType type, Dist<double[]> dist ) {
		List<TreeNode> l = new ArrayList<>(tree);
		Collections.sort(l, new Comparator<TreeNode>() {
			@Override
			public int compare(TreeNode o1, TreeNode o2) {
				return Integer.compare(o1.age, o2.age);
			}
		});
		
		Map<TreeNode,Set<double[]>> contents = new HashMap<>();
		for( TreeNode tn : tree )
			contents.put(tn, getContents(tn));
		
		Map<double[],Set<double[]>> m = new HashMap<>();	
		for( TreeNode tn : l ) { // iterate by age, starting from 0
			if( tn.age == 0 ) { // s contains only one element
				m.put(contents.get(tn).iterator().next(), new HashSet<>());
				continue;
			}

			double[] bestA = null, bestB = null;

			// always shortest??
			if( type == HierarchicalClusteringType.complete_linkage || type == HierarchicalClusteringType.single_linkage ) {
				double minDist = Double.MAX_VALUE;
				for (double[] a : contents.get( tn.children.get(0) ) )
					for (double[] b : contents.get( tn.children.get(1) ) )
						if (cm.get(a).contains(b)) {
							double d = dist.dist(a, b);
							if (d < minDist) {
								minDist = d;
								bestA = a;
								bestB = b;
							}
						}
			} else {
				double minDist = Double.MAX_VALUE;
				for (double[] a : contents.get( tn.children.get(0) ) )
					for (double[] b : contents.get( tn.children.get(1) ) )
						if ( cm.get(a).contains(b) ) {
							double d = dist.dist(a, b);
							if (d < minDist) {
								minDist = d;
								bestA = a;
								bestB = b;
							}
						}
			}
			m.get(bestA).add(bestB);
			m.get(bestB).add(bestA);
		}		
		return m;
	}
		
	public static List<Set<double[]>> cutTreeREDCAP( Collection<TreeNode> tree, Map<double[],Set<double[]>> cm, HierarchicalClusteringType type, int numCluster, Dist<double[]> dist ) {
		Map<double[],Set<double[]>> spaningTree = Clustering.toREDCAPSpanningTree(tree,cm,type,dist); // its bidirectional
		return cutTreeREDCAP( spaningTree, numCluster, dist);
	}
	
	public static List<Set<double[]>> cutTreeREDCAP( Map<double[],Set<double[]>> spaningTree, int numCluster, Dist<double[]> dist ) {
		
		Map<Map<double[], Set<double[]>>,Double> subs = new HashMap<>();
		for( Map<double[],Set<double[]>> sub : GraphUtils.getSubGraphs(spaningTree) )
			subs.put(sub, DataUtils.getSumOfSquares(GraphUtils.getNodes(sub), dist ));
		
		while( subs.size() != numCluster ) {
			// get subtree with largest cluster-SS
			Map<double[],Set<double[]>> bestT = null;
			double cost = Double.MIN_VALUE;
			for( Entry<Map<double[],Set<double[]>>,Double> e : subs.entrySet() ) 
				if( e.getValue() > cost ) {
					cost = e.getValue();
					bestT = e.getKey();
				}
			subs.remove(bestT);
						
			// get best cut of tree
			double[] bestA = null, bestB = null;
			double bestInc = Double.MIN_VALUE;
			Set<double[]> nodesBestT = GraphUtils.getNodes( bestT );
			for( double[] a : new ArrayList<double[]>(bestT.keySet() ) )
				for( double[] b : new ArrayList<double[]>(bestT.get(a) ) ) {
					if( a.hashCode() < b.hashCode() ) // just one direction is enough
						continue;
					
					bestT.get(a).remove(b);
					bestT.get(b).remove(a);
						
					Set<double[]> nodesA = GraphUtils.getNodes( GraphUtils.getSubGraphOf(bestT, a) );
					
					// NOTE: Assumes that treeA is complementary to treeB (which should always be the case)
					//Set<double[]> nodesB = GraphUtils.getNodes( GraphUtils.getSubGraphOf(bestT, b) );
					Set<double[]> nodesB = new HashSet<double[]>(nodesBestT);
					nodesB.removeAll(nodesA);
										
					double inc = cost - DataUtils.getSumOfSquares( nodesA, dist ) - DataUtils.getSumOfSquares( nodesB, dist );
					if( inc > bestInc ) {
						bestInc = inc;
						bestA = a;
						bestB = b;
					}
							
					bestT.get(a).add(b);
					bestT.get(b).add(a);
				}
			
			// do the cut, replace tree by subtrees
			bestT.get(bestA).remove(bestB);
			bestT.get(bestB).remove(bestA);
						
			Map<double[], Set<double[]>> subA = GraphUtils.getSubGraphOf(bestT, bestA);
			Map<double[], Set<double[]>> subB = GraphUtils.getSubGraphOf(bestT, bestB);
						
			subs.put(subA,DataUtils.getSumOfSquares(GraphUtils.getNodes(subA), dist ));
			subs.put(subB,DataUtils.getSumOfSquares(GraphUtils.getNodes(subB), dist ));
		}
				
		// to cluster
		List<Set<double[]>> cluster = new ArrayList<Set<double[]>>();
		for( Map<double[], Set<double[]>> s : subs.keySet() ) 
			cluster.add(GraphUtils.getNodes(s));
		return cluster;
	}
						
	// not connected
	public static List<TreeNode> getHierarchicalClusterTree(List<double[]> samples, Dist<double[]> dist, HierarchicalClusteringType type) {
		List<TreeNode> l = new ArrayList<TreeNode>();
		for( double[] d : samples ) {
			TreeNode tn = new TreeNode();
			tn.age = 0;
			tn.cost = 0;
			tn.contents = new HashSet<>();
			tn.contents.add(d);
			l.add(tn);
		}	
		return getHierarchicalClusterTree(l, null, dist, type);
	}
	
	// connected
	public static List<TreeNode> getHierarchicalClusterTree(Map<double[], Set<double[]>> cm, Dist<double[]> dist, HierarchicalClusteringType type) {
		Set<double[]> s = new HashSet<double[]>();
		for( Entry<double[],Set<double[]>> e : cm.entrySet() ) {
			s.add(e.getKey());
			s.addAll(e.getValue());
		}
				
		List<TreeNode> l = new ArrayList<TreeNode>();
		for( double[] d : s ) {
			TreeNode tn = new TreeNode();
			tn.age = 0;
			tn.cost = 0;
			tn.contents = new HashSet<>();
			tn.contents.add(d);
			l.add(tn);
		}
		
		Map<TreeNode,Set<TreeNode>> ncm = new HashMap<>();
		for( TreeNode tnA : l ) {
			double[] a = tnA.contents.iterator().next();
			
			Set<TreeNode> cs = new HashSet<>();
			for( double[] b : cm.get(a) )
				for( TreeNode tnB : l )
					if( tnB.contents.contains(b) )
						cs.add(tnB);
			ncm.put(tnA, cs);
		}
				
		return getHierarchicalClusterTree( l, ncm, dist, type);
	}
	
	public static List<TreeNode> getHierarchicalClusterTree( List<TreeNode> leafLayer, Map<TreeNode,Set<TreeNode>> cm, Dist<double[]> dist, HierarchicalClusteringType type) {
						
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
				
		List<TreeNode> tree = new ArrayList<>();
		Map<TreeNode,Set<double[]>> curLayer = new HashMap<>();
		
		Map<TreeNode, Double> ssCache = new HashMap<TreeNode, Double>();
		Map<TreeNode, Map<TreeNode,Double>> unionCache = new HashMap<>();
					
		int length = getContents(leafLayer.get(0)).iterator().next().length;
		int age = 0;
		for( TreeNode tn : leafLayer ) {
			
			age = Math.max( age, tn.age );
			tree.add(tn);
			
			Set<double[]> content = getContents(tn);
			curLayer.put(tn, content);
			ssCache.put(tn, DataUtils.getSumOfSquares(content, dist));
		}
						
		// copy of connected map
		Map<TreeNode, Set<TreeNode>> connected = null;
		if (cm != null) {
			connected = new HashMap<TreeNode, Set<TreeNode>>();
			for( Entry<TreeNode,Set<TreeNode>> e : cm.entrySet() )
				connected.put(e.getKey(),new HashSet<TreeNode>(e.getValue()));
		}
		
		while (curLayer.size() > 1 ) {
			TreeNode c1 = null, c2 = null;
			double sMin = Double.MAX_VALUE;
			
			List<TreeNode> cl = new ArrayList<>(curLayer.keySet());
			for (int i = 0; i < cl.size() - 1; i++) {
				TreeNode l1 = cl.get(i);

				for (int j = i + 1; j < cl.size(); j++) {
					TreeNode l2 = cl.get(j);
					
					if( connected != null && ( !connected.containsKey(l1) || !connected.get(l1).contains(l2) ) ) // disjoint
						continue;
																				
					double s = Double.NaN;
					if (HierarchicalClusteringType.ward == type) {																														
						// get error sum of squares
						if (!unionCache.containsKey(l1) || !unionCache.get(l1).containsKey(l2)) {
																
							// calculate mean and sum of squares, slightly faster than actually forming a union
							double[] mean = new double[length];
							for (int l = 0; l < length; l++) {
								for (double[] d : curLayer.get(l1) )
									mean[l] += d[l];
								for (double[] d : curLayer.get(l2) )
									mean[l] += d[l];
							}
														
							for (int l = 0; l < length; l++)
								mean[l] /= curLayer.get(l1).size()+curLayer.get(l2).size();
							
							double ssUnion = 0;
							for( double[] d : curLayer.get(l1) ) {
								double di = dist.dist(mean, d);
								ssUnion += di * di;
							}
							for( double[] d : curLayer.get(l2) ) {
								double di = dist.dist(mean, d);
								ssUnion += di * di;
							}
							
							if (!unionCache.containsKey(l1))
								unionCache.put( l1, new HashMap<TreeNode, Double>() );
							unionCache.get(l1).put(l2, ssUnion);
						}													
						s = unionCache.get(l1).get(l2) - ( ssCache.get(l1) + ssCache.get(l2) );	
					} else if (HierarchicalClusteringType.single_linkage == type) {
						s = Double.MAX_VALUE;
						for (double[] d1 : curLayer.get(l1)) 
							for (double[] d2 : curLayer.get(l2)) 
								s = Math.min(s, dist.dist(d1, d2) );				
					} else if (HierarchicalClusteringType.complete_linkage == type) {
						s = Double.MIN_VALUE;
						for (double[] d1 : curLayer.get(l1))
							for (double[] d2 : curLayer.get(l2))
								s = Math.max(s, dist.dist(d1, d2) );
					} else if (HierarchicalClusteringType.average_linkage == type) {
						s = 0;
						for (double[] d1 : curLayer.get(l1)) 
							for (double[] d2 : curLayer.get(l2)) 
								s += dist.dist(d1, d2);
						s /= (curLayer.get(l1).size() * curLayer.get(l2).size());
					}
					if (s < sMin) {
						c1 = l1;
						c2 = l2;
						sMin = s;
					}
				}
			}
			
			if( c1 == null && c2 == null ) { // no connected clusters present anymore
				log.warn("only non-connected clusters present! "+curLayer.size() );
				return tree;
			}
			
			// create merge node, remove c1,c2		
			TreeNode mergeNode = new TreeNode();
			Set<double[]> union = new FlatSet<double[]>(); 
			union.addAll(curLayer.remove(c1));
			union.addAll(curLayer.remove(c2));	
			mergeNode.age = ++age;
			mergeNode.children = Arrays.asList(new TreeNode[]{ c1, c2 });
			mergeNode.cost = sMin;
			
			// update cache
			if( type == HierarchicalClusteringType.ward ) {
				ssCache.remove(c1);
				ssCache.remove(c2);
				ssCache.put( mergeNode, unionCache.get(c1).get(c2) );
				unionCache.remove(c1);
			} 
									
			// add nodes
			curLayer.put(mergeNode,union);
			tree.add(mergeNode);
			
			// update connected map
			if( connected != null ) {
				// 1. merge values of c1 and c2 and put union
				Set<TreeNode> ns = connected.remove(c1);
				ns.addAll( connected.remove(c2) );
				connected.put(mergeNode, ns);
				
				// 2. replace all values c1,c2 by merged node
				for( Set<TreeNode> s : connected.values() ) {
					if( s.contains(c1) || s.contains(c2)) {
						s.remove(c1);
						s.remove(c2);
						s.add(mergeNode);
					}
				}
			}

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
		DataUtils.transform(samples, fa, Transform.zScore);
		final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");
		final Dist<double[]> dist = new EuclideanDist(fa);
								
		int nrCluster = 7;
		long time = System.currentTimeMillis();
		List<TreeNode> tree = Clustering.getHierarchicalClusterTree(cm, dist, HierarchicalClusteringType.ward);
		List<Set<double[]>> ct = Clustering.cutTree( tree, nrCluster);
		log.debug("Nr cluster: "+ct.size());
		log.debug("Within sum of squares: " + DataUtils.getWithinSumOfSquares(ct, dist)+", took: "+(System.currentTimeMillis()-time)/1000.0);		
		Drawer.geoDrawCluster(ct, samples, geoms, "output/clustering.png", true);
	}
}
