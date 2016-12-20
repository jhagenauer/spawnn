package spawnn.utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import spawnn.dist.Dist;

public class GraphUtils {
	
	public static Map<double[],Set<double[]>> getUnweightedGraph( Map<double[],Map<double[],Double>> graph ) {
		Map<double[],Set<double[]>> ng = new HashMap<double[],Set<double[]>>();
				for( double[] a : graph.keySet() ) 
			ng.put(a, graph.get(a).keySet() );
		return ng;
	}

	public static Map<double[],Map<double[],Double>> getWeightedGraph( Map<double[],Set<double[]>> graph, Dist<double[]> d) {
		Map<double[],Map<double[],Double>> ng = new HashMap<double[],Map<double[],Double>>();
		for( double[] a : graph.keySet() ) {
			Map<double[],Double> m = new HashMap<double[],Double>();
			for( double[] b : graph.get(a) )
				m.put(b, d.dist(a, b));
			ng.put(a, m);
		}
		return ng;
	}

	public static <T> Set<T> getNodes(Map<T, Set<T>> cm) {
		Set<T> nodes = new HashSet<T>(cm.keySet());
		for (T a : cm.keySet())
			nodes.addAll(cm.get(a));
		return nodes;
	}

	public static <T> List<Map<T, Set<T>>> getSubGraphs(Map<T, Set<T>> cm) {
		List<Map<T, Set<T>>> subs = new ArrayList<Map<T, Set<T>>>();
		Set<T> allNodes = getNodes(cm);
	
		while (!allNodes.isEmpty()) {
	
			// get first non-visited node
			T initNode = allNodes.iterator().next();
			allNodes.remove(initNode);
	
			Map<T, Set<T>> sg = GraphUtils.getSubGraphOf(cm, initNode);
			allNodes.removeAll(getNodes(sg));
			subs.add(sg);
		}
		return subs;
	}

	static <T> Map<T, Set<T>> getSubGraphOf(Map<T, Set<T>> cm, T initNode) {
		Map<T, Set<T>> visited = new HashMap<T, Set<T>>(); // expanded/subgraph nodes
		Set<T> open = new HashSet<T>();
		open.add(initNode);
	
		while (!open.isEmpty()) {
			T cur = open.iterator().next();
			open.remove(cur);
			visited.put(cur,new HashSet<T>() );
	
			for (T nb : cm.get(cur)) {
				visited.get(cur).add(nb);
				
				if( nb != cur && !visited.containsKey(nb) )
					open.add(nb);
			}
		}
		return visited;
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

	// assumes that cm is directed and is fully connected! Prim's algorithm
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
	
	public static List<Entry<double[],Double>> getDijkstraShortestPath( Map<double[],Map<double[],Double>> graph, double[] from, double[] to ) { 
		Set<double[]> openList = new HashSet<double[]>();
		Map<double[],Double> distMap = new HashMap<double[],Double>();
		Map<double[],Entry<double[],Double>> precessors = new HashMap<double[],Entry<double[],Double>>();
		
		openList.add( from );
		distMap.put( from, 0.0 );
		
		while( !openList.isEmpty() ) {
			// find nearest 
			double min = Double.POSITIVE_INFINITY;
			double[] curNode = null;
			for( double[] v : openList ) {
				if( distMap.get(v) < min ) {
					min = distMap.get(v);
					curNode = v;
				}
			}
			openList.remove(curNode);
						
			if( curNode.equals(to) ) { // path found, reconstruct path
				LinkedList<Entry<double[],Double>> path = new LinkedList<Entry<double[],Double>>();
				double[] n = curNode;
				while( !n.equals(from) ) {
					Entry<double[],Double> pre = precessors.get(n);
					path.addFirst( new AbstractMap.SimpleEntry<double[],Double>(n, pre.getValue() ) );
					n = pre.getKey();
				} 
				return path;
			}
			
			for( Entry<double[],Double> e : graph.get(curNode).entrySet() ) {
				double[] nb = e.getKey();				
				double d = min + e.getValue();
				if( !distMap.containsKey(nb) || d < distMap.get(nb) ) {
					distMap.put( nb, d );
					precessors.put( nb, new AbstractMap.SimpleEntry<double[],Double>(curNode,e.getValue() ) );
					openList.add(nb);
				} 
			}
		}
		return null;
	}

	public static Map<double[], Double> getShortestDists(Map<double[], Map<double[], Double>> graph, double[] from) {
		Set<double[]> openList = new HashSet<double[]>();
		Map<double[], Double> distMap = new HashMap<double[], Double>();
		Map<double[], Entry<double[], Double>> precessors = new HashMap<double[], Entry<double[], Double>>();
	
		openList.add(from);
		distMap.put(from, 0.0);
	
		while (!openList.isEmpty()) {
			// find nearest
			double min = Double.POSITIVE_INFINITY;
			double[] curNode = null;
			for (double[] v : openList) {
				if (distMap.get(v) < min) {
					min = distMap.get(v);
					curNode = v;
				}
			}
			openList.remove(curNode);
	
			for (Entry<double[], Double> e : graph.get(curNode).entrySet()) {
				double[] nb = e.getKey();
				double d = min + e.getValue();
				if (!distMap.containsKey(nb) || d < distMap.get(nb)) {
					distMap.put(nb, d);
					precessors.put(nb, new AbstractMap.SimpleEntry<double[], Double>(curNode, e.getValue()));
					openList.add(nb);
				}
			}
		}
		return distMap;
	}

	public static void writeContiguityMap( Map<double[], Set<double[]>> cm, List<double[]> samples, String fn ) {
		BufferedWriter bw = null;
		try {
			 bw = new BufferedWriter( new FileWriter(fn ) );
			 bw.write("id1,id2\n");
			 for( double[] a : cm.keySet() ) 
				 for( double[] b : cm.get(a) ) 
					 bw.write(samples.indexOf(a)+","+samples.indexOf(b)+"\n");
				 
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				bw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}	
	}
}
