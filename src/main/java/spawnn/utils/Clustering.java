package spawnn.utils;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import javax.imageio.ImageIO;

import org.apache.log4j.Logger;
import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.geotools.geometry.jts.ReferencedEnvelope;
import org.geotools.map.FeatureLayer;
import org.geotools.map.MapContent;
import org.geotools.renderer.GTRenderer;
import org.geotools.renderer.lite.StreamingRenderer;
import org.geotools.styling.SLD;
import org.geotools.styling.StyleBuilder;

import regionalization.RegionUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.LineString;

public class Clustering {

	private static Logger log = Logger.getLogger(Clustering.class);

	// TODO: Sucks big time, why?
	public static Map<double[], Set<double[]>> kMedoidsPAM(Collection<double[]> samples, int num, Dist<double[]> dist) {
		Random r = new Random();

		Map<double[], Set<double[]>> medoidMap = new HashMap<double[], Set<double[]>>();

		// 1. random init
		while (medoidMap.size() < num) {// random init centroids
			for (double[] s : samples)
				if (r.nextDouble() < 1.0 / samples.size() && !medoidMap.containsKey(s)) {
					medoidMap.put(s, new HashSet<double[]>());
					break;
				}
		}

		double actualCost = Double.MAX_VALUE;
		while (true) {

			// 2. clear, and add closest samples to medoids
			for (Set<double[]> s : medoidMap.values())
				s.clear();

			for (double[] s : samples) {
				double bestDist = Double.MAX_VALUE;
				double[] bestMedoid = null;

				if (medoidMap.containsKey(s)) // if s is medoid
					continue;

				for (double[] medoid : medoidMap.keySet()) {
					if (bestMedoid == null || dist.dist(s, medoid) < bestDist) {
						bestMedoid = medoid;
						bestDist = dist.dist(s, medoid);
					}
				}
				medoidMap.get(bestMedoid).add(s);
			}

			// 4. for each
			Map<double[], Set<double[]>> workingCopy = new HashMap<double[], Set<double[]>>();
			for (Entry<double[], Set<double[]>> e : medoidMap.entrySet())
				workingCopy.put(e.getKey(), new HashSet<double[]>(e.getValue()));

			double bestCost = Double.MAX_VALUE;
			double[] bestMedoid = null, bestSwap = null, bestSwapSetMedoid = null;

			for (double[] medoid : medoidMap.keySet()) { // all medoids
				for (double[] swapSetMedoid : medoidMap.keySet()) {
					for (double[] s : medoidMap.get(swapSetMedoid)) { // all
																		// data

						// swap
						workingCopy.get(swapSetMedoid).remove(s);
						workingCopy.get(swapSetMedoid).add(medoid);
						Set<double[]> old = workingCopy.remove(medoid);
						workingCopy.put(s, old);

						double cost = DataUtils.getMeanQuantizationError(workingCopy, dist);
						if (cost < bestCost) {
							bestMedoid = medoid;
							bestSwap = s;
							bestSwapSetMedoid = swapSetMedoid;
							bestCost = cost;
						}

						// swap back
						workingCopy.remove(s);
						workingCopy.put(medoid, old);
						workingCopy.get(swapSetMedoid).remove(medoid);
						workingCopy.get(swapSetMedoid).add(s);

					}
				}
			}

			if (bestCost < actualCost) {
				// System.out.println(bestCost);
				// swap
				medoidMap.get(bestSwapSetMedoid).remove(bestSwap);
				medoidMap.get(bestSwapSetMedoid).add(bestMedoid);
				Set<double[]> old = medoidMap.remove(bestMedoid);
				medoidMap.put(bestSwap, old);

				actualCost = bestCost;
			} else {
				// System.out.println("break");
				break;
			}

		}

		// add medoids to form clusters
		for (double[] d : medoidMap.keySet())
			medoidMap.get(d).add(d);

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

	public static List<Set<double[]>> cutTree( Map<Set<double[]>,TreeNode> tree, int numCluster ) {
		Set<TreeNode> done = new HashSet<TreeNode>();
		Map<TreeNode,Set<double[]>> cutTree = new HashMap<TreeNode,Set<double[]>>();
		
		// init with leafs
		for( Map.Entry<Set<double[]>, TreeNode> e : tree.entrySet() ) {
			if( e.getValue().children == null ) {
				cutTree.put(e.getValue(),e.getKey());
				done.add(e.getValue());
			}
		}
					
		while( cutTree.size() > numCluster ) {
			
			Map.Entry<Set<double[]>, TreeNode> bestEntry = null;
			for( Map.Entry<Set<double[]>, TreeNode> e : tree.entrySet() ) {
				if( done.contains(e.getValue()) ) // already merged
					continue;
				
				// children present in tree
				boolean childenInTree = true;
				for( TreeNode child : e.getValue().children )
					if( !cutTree.containsKey(child) )
						childenInTree = false;
				if( !childenInTree )
					continue;
				
				if( !Double.isNaN(e.getValue().sumOfSquares) && ( bestEntry == null || e.getValue().sumOfSquares < bestEntry.getValue().sumOfSquares ) ) 
					bestEntry = e;								
			}
			
			if( bestEntry == null ) {
				log.warn("Cannot cut tree any further.");
				break;
			}
									
			for( TreeNode child : bestEntry.getValue().children ) {
				cutTree.remove(child);
				done.add(child);
			}			
			cutTree.put(bestEntry.getValue(), bestEntry.getKey() );
		}
		
		// tree to cluster
		List<Set<double[]>> clusters = new ArrayList<Set<double[]>>();
		for( Set<double[]> s : cutTree.values() )
			clusters.add(s);
		
		return clusters;
	}
	
	public static class TreeNode {
		double sumOfSquares = 0;
		TreeNode children[] = null;
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
			
			@SuppressWarnings("unchecked")
			@Override
			public boolean addAll( Collection<? extends T> c ) {
				hashCode += ((FlatSet<double[]>)c).hashCode;
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
						
		// init
		for (double[] d : samples) {
			Set<double[]> l = new FlatSet<double[]>();
			l.add(d);
			
			TreeNode cn = new TreeNode();
			cn.sumOfSquares = 0;
			tree.put(l,cn);
			
			leafLayer.add(l);
			ssCache.put(l, 0.0);
		}
		
		// init connected map
		Map<Set<double[]>, Set<Set<double[]>>> connected = null;
		if( cm != null ) {
			connected = new HashMap<Set<double[]>, Set<Set<double[]>>>();
			for( Set<double[]> a : leafLayer ) {
				connected.put(a, new HashSet<Set<double[]>>());
				for( Set<double[]> b : leafLayer ) {
					if( a != b && cm.get(a.iterator().next()).contains(b.iterator().next())) {
						connected.get(a).add(b);
						if( !connected.containsKey(b) )
							connected.put(b, new HashSet<Set<double[]>>());
						connected.get(b).add(a);
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
					
					if( connected != null && ( !connected.containsKey(l1) || !connected.get(l1).contains(l2) ) ) // disjoint
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
				
				//get ss
				for( Set<double[]> s : leafLayer )
					if( s == union )
						ss += unionCache.get(c1).get(c2);
					else
						ss += ssCache.get(s);
				
				// update caches
				ssCache.remove(c1);
				ssCache.remove(c2);
				ssCache.put( union, unionCache.get(c1).get(c2) );
				
				// update connected map
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

				unionCache.remove(c1);
			} else
				ss = DataUtils.getWithinClusterSumOfSuqares(leafLayer, dist); // expensive
						
			TreeNode cn = new TreeNode();
			cn.sumOfSquares = ss;
			cn.children = new TreeNode[]{ tree.get(c1), tree.get(c2) };
			tree.put(union, cn);
			
			//log.debug( leafLayer.size()+", ss:"+ss);
		}
		return tree;
	}
	
	public static Map<double[], Set<double[]>> deriveQueenContiguitiyMap(List<double[]> samples, List<Geometry> geoms) {
		Map<double[], Set<double[]>> cm = new HashMap<double[], Set<double[]>>();

		for (int i = 0; i < samples.size(); i++) {
			double[] a = samples.get(i);
			Geometry ag = geoms.get(i);

			cm.put(a, new HashSet<double[]>());

			for (int j = 0; j < samples.size(); j++) {
				double[] b = samples.get(j);
				Geometry bg = geoms.get(j);

				if (bg.touches(ag) || bg.intersects(ag))
					cm.get(a).add(b);
			}
		}
		return cm;
	}

	// assumes that cm is not directed and connected! Prim's algorithm
	public static Map<double[], Set<double[]>> getMinimumSpanningTree(Map<double[], Set<double[]>> cm, Dist<double[]> dist) {
		Map<double[], Set<double[]>> mst = new HashMap<double[], Set<double[]>>();

		Set<double[]> added = new HashSet<double[]>();
		added.add(cm.keySet().iterator().next());

		while (added.size() != cm.size()) { // maybe critical if undirected

			double[] bestA = null, bestB = null;
			double minDist = Double.MAX_VALUE;

			for (double[] a : added) {
				for (double[] b : cm.get(a)) {
					if (added.contains(b))
						continue;

					double d = dist.dist(a, b);
					if (d < minDist) {
						minDist = d;
						bestA = a;
						bestB = b;
					}
				}
			}

			// add connections to both directions
			if (!mst.containsKey(bestA))
				mst.put(bestA, new HashSet<double[]>());
			mst.get(bestA).add(bestB);

			if (!mst.containsKey(bestB))
				mst.put(bestB, new HashSet<double[]>());
			mst.get(bestB).add(bestA);

			added.add(bestB);
		}

		return mst;
	}

	public static boolean isUndirected(Map<double[], Set<double[]>> cm) {
		boolean undirected = true;

		for (double[] a : cm.keySet()) {
			for (double[] b : cm.get(a)) {
				if (!cm.containsKey(b) || !cm.get(b).contains(a))
					undirected = false;
			}
		}
		return undirected;
	}

	public static Map<double[], Set<double[]>> getUndirectedGraph(Map<double[], Set<double[]>> cm) {
		Map<double[], Set<double[]>> undirected = new HashMap<double[], Set<double[]>>();

		for (double[] a : cm.keySet()) {
			if (!undirected.containsKey(a))
				undirected.put(a, new HashSet<double[]>());
			for (double[] b : cm.get(a)) {
				undirected.get(a).add(b);

				if (!undirected.containsKey(b))
					undirected.put(b, new HashSet<double[]>());
				undirected.get(b).add(a);
			}
		}
		return undirected;
	}

	private static Map<double[], Set<double[]>> getSubGraphOf(Map<double[], Set<double[]>> cm, double[] initNode) {
		Map<double[], Set<double[]>> visited = new HashMap<double[], Set<double[]>>(); // expanded/subgraph nodes
		Set<double[]> open = new HashSet<double[]>();
		open.add(initNode);

		while (!open.isEmpty()) {
			double[] cur = open.iterator().next();
			open.remove(cur);
			visited.put(cur,new HashSet<double[]>() );

			for (double[] nb : cm.get(cur)) {
				visited.get(cur).add(nb);
				
				if( nb != cur && !visited.containsKey(nb) )
					open.add(nb);
			}
		}
		return visited;
	}
	
	public static List<Map<double[], Set<double[]>>> getSubGraphs(Map<double[], Set<double[]>> cm) {
		List<Map<double[], Set<double[]>>> subs = new ArrayList<Map<double[], Set<double[]>>>();
		Set<double[]> allNodes = getNodes(cm);

		while (!allNodes.isEmpty()) {

			// get first non-visited node
			double[] initNode = allNodes.iterator().next();
			allNodes.remove(initNode);

			Map<double[], Set<double[]>> sg = getSubGraphOf(cm, initNode);
			allNodes.removeAll(getNodes(sg));
			subs.add(sg);
		}
		return subs;
	}

	public static void geoDrawConnectivityMap(Map<double[], Set<double[]>> cm, int[] ga, String fn) {
		// draw
		try {
			SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
			typeBuilder.setName("lines");
			typeBuilder.add("the_geom", LineString.class);

			SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());

			StyleBuilder sb = new StyleBuilder();
			MapContent map = new MapContent();
			ReferencedEnvelope maxBounds = null;

			GeometryFactory gf = new GeometryFactory();
			DefaultFeatureCollection fc = new DefaultFeatureCollection();
			for (double[] a : cm.keySet()) {
				for (double[] b : cm.get(a)) {
					Geometry g = gf.createLineString(new Coordinate[] { new Coordinate(a[ga[0]], a[ga[1]]), new Coordinate(b[ga[0]], b[ga[1]]) });
					featureBuilder.add(g);
					fc.add(featureBuilder.buildFeature("" + fc.size()));
				}
			}

			maxBounds = fc.getBounds();
			map.addLayer(new FeatureLayer(fc, SLD.wrapSymbolizers(sb.createLineSymbolizer(Color.BLACK))));

			GTRenderer renderer = new StreamingRenderer();
			RenderingHints hints = new RenderingHints(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			hints.put(RenderingHints.KEY_ALPHA_INTERPOLATION, RenderingHints.VALUE_ALPHA_INTERPOLATION_QUALITY);
			hints.put(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_SPEED);

			renderer.setMapContent(map);

			Rectangle imageBounds = null;
			try {
				double heightToWidth = maxBounds.getSpan(1) / maxBounds.getSpan(0);
				int imageWidth = 1024*5;

				imageBounds = new Rectangle(0, 0, imageWidth, (int) Math.round(imageWidth * heightToWidth));
				// imageBounds = new Rectangle( 0, 0, mp.getWidth(), (int)
				// Math.round(mp.getWidth() * heightToWidth));

				BufferedImage image = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_RGB);
				Graphics2D gr = image.createGraphics();
				gr.setPaint(Color.WHITE);
				gr.fill(imageBounds);

				renderer.paint(gr, imageBounds, maxBounds);

				ImageIO.write(image, "png", new File(fn));
				image.flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
			map.dispose();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static Set<double[]> getNodes(Map<double[], Set<double[]>> cm) {
		Set<double[]> nodes = new HashSet<double[]>(cm.keySet());
		for (double[] a : cm.keySet())
			nodes.addAll(cm.get(a));
		return nodes;
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
			for (Map<double[], Set<double[]>> sg : getSubGraphs(mst)) {

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

					Set<double[]> sgA = getNodes(getSubGraphOf(sg, e.a));
					Set<double[]> sgB = getNodes(getSubGraphOf(sg, e.b));
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
							Set<double[]> s = getNodes(getSubGraphOf(sg, e.a));
							sgCache.put(s, DataUtils.getWithinClusterSumOfSuqares(s, dist));
						}

						// remove edge
						sg.get(e.a).remove(e.b);
						sg.get(e.b).remove(e.a);

						Set<double[]> nodesA = getNodes(getSubGraphOf(sg, e.a));
						Set<double[]> nodesB = getNodes(getSubGraphOf(sg, e.b));

						double ssdT_a = DataUtils.getWithinClusterSumOfSuqares(nodesA, dist);
						double ssdT_b = DataUtils.getWithinClusterSumOfSuqares(nodesB, dist);

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
		for (Map<double[], Set<double[]>> sg : getSubGraphs(mst))
			clusters.add(getNodes(sg));

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
		{
		System.out.println("skater:");
		long time = System.currentTimeMillis();
		Map<double[], Set<double[]>> mst = getMinimumSpanningTree(cm, dist);
		System.out.println("mst took: " + (System.currentTimeMillis() - time) / 1000.0);
		time = System.currentTimeMillis();
		List<Set<double[]>> cluster = skater(mst, nrCluster - 1, dist, 1);
		System.out.println("skater took: " + (System.currentTimeMillis() - time) / 1000.0);
		System.out.println("WCSS: " + DataUtils.getWithinClusterSumOfSuqares(cluster, dist));
		}
		
		{
		System.out.println("ward:");
		long time = System.currentTimeMillis();
		Map<Set<double[]>,TreeNode> tree = Clustering.getHierarchicalClusterTree(samples, cm, dist, HierarchicalClusteringType.ward);
		System.out.println("tree took: " + (System.currentTimeMillis() - time) / 1000.0);
		time = System.currentTimeMillis();
		List<Set<double[]>> cluster = Clustering.cutTree( tree, nrCluster);
		System.out.println("cut took: " + (System.currentTimeMillis() - time) / 1000.0);
		System.out.println("WCSS: " + DataUtils.getWithinClusterSumOfSuqares(cluster, dist));
		}	
	}
}
