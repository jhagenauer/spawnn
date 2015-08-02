package spawnn.utils;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.collections15.Transformer;
import org.apache.log4j.Logger;

import edu.uci.ics.jung.graph.UndirectedSparseGraph;
import edu.uci.ics.jung.io.GraphIOException;
import edu.uci.ics.jung.io.graphml.EdgeMetadata;
import edu.uci.ics.jung.io.graphml.GraphMLReader2;
import edu.uci.ics.jung.io.graphml.GraphMetadata;
import edu.uci.ics.jung.io.graphml.HyperEdgeMetadata;
import edu.uci.ics.jung.io.graphml.NodeMetadata;

public class GraphClustering {

	private static Logger log = Logger.getLogger(GraphClustering.class);
	
	public static Map<double[],Map<double[],Double>> toWeightedGraph( Map<double[],Set<double[]>> graph ) {
		Map<double[],Map<double[],Double>> ng = new HashMap<double[],Map<double[],Double>>();
		for( double[] a : graph.keySet() ) {
			Map<double[],Double> m = new HashMap<double[],Double>();
			for( double[] b : graph.get(a) )
				m.put(b, 1.0);
			ng.put(a, m);
		}
		return ng;
	}
		
	public static <V,E> double modularity ( Map<double[],Map<double[],Double>> graph, Map<double[],Integer> membership ) {
        double sum = 0;
        double m2 = 0;
        for( Map<double[],Double> m : graph.values() )
        	for( double weight : m.values() )
        		m2 += weight;
                
        for( double[] v1 : graph.keySet() ) {
        	double ki = 0;
        	
        	for( double weight : graph.get(v1).values() )
        		ki += weight;
        	
        	for( double[] v2 : graph.keySet() ) {
        		if( membership.get(v1) != membership.get(v2) ) 
        			continue;
        		
        		double kj = 0;
        		for( double weight : graph.get(v2).values() )
        			kj += weight;
        		
        		double weight = 0;
        		if( graph.get(v1).containsKey(v2) ) 
        			weight = graph.get(v1).get(v2);
        		sum += weight - ki*kj/m2;
        	}	
        }
        return sum/m2;
	}
	
	public static Collection<Set<double[]>> modulMapToCluster( Map<double[],Integer> map ) {
		Map<Integer,Set<double[]>> m = new HashMap<Integer,Set<double[]>>();
		for( double[] d : map.keySet() ) {
			int i = map.get(d);
			if( !m.containsKey(i) )
				m.put( i, new HashSet<double[]>() );
			m.get(i).add(d);
		}
		return m.values();
	}
	
	// if nrCluster <= 0, return best clusters, otherwise return desired number of clusters
	public static Map<double[],Integer> greedyOptModularity( Map<double[],Map<double[],Double>> graph, int rndRestarts, final int nrCluster ) {		
			
		Map<double[],Integer> bestMap = null; // best results
		double bestModularity = 0;
		for( int i = 0; i < rndRestarts; i++ ) {
			
			List<double[]> vertices = new ArrayList<double[]>(graph.keySet());
			Collections.shuffle(vertices);
			
			// init module assignment
			Map<double[],Integer> curMap = new HashMap<double[],Integer>();
			int k = 0;
			for( double[] a : vertices )
				curMap.put(a, k++); 
			
			// init map of connections between modules, no self-loops, 
			Map<Integer,Set<Integer>> curConMap = new HashMap<Integer,Set<Integer>>();
			for( double[] v : vertices ) {
				int c = curMap.get(v);
				Set<Integer> s = new HashSet<Integer>();
				for( double[] nb : graph.get(v).keySet() ) {
					int cnb = curMap.get(nb);
					if( c != cnb // no self-loops
						&& !( curConMap.containsKey(cnb) && curConMap.get(cnb).contains(c) ) ) // only one direction
						s.add( cnb );
				}
				curConMap.put( c, s);
			}			
						 				
			int merges = 0;
			while( nrCluster <= 0 || curMap.size() - merges++ > nrCluster ) {
				double bestInc = 0;
				int bestModulV = -1, bestModulNB = -1;
								
				for( int modulV : curConMap.keySet() ) {
					for( int modulNB : curConMap.get(modulV) ) { 
				
						double modBefore = modularity(graph, curMap);
								
						// merge					
						Set<double[]> s = new HashSet<double[]>();
						for( double[] d : curMap.keySet() )
							if( curMap.get(d) == modulNB )
								s.add(d);
						
						for( double[] d : s )
							curMap.put(d, modulV);
									
						double inc = modularity(graph, curMap) - modBefore; 
						
						if( bestModulV < 0 || inc > bestInc ) {
							bestInc = inc;
							bestModulV = modulV;
							bestModulNB = modulNB;
						}
						
						// restore old memberships
						for( double[] d : s )
							curMap.put(d, modulNB);
					}
				}
				
				if( ( bestModulV < 0 || bestModulNB < 0 ) // no further clustering possible
					|| ( nrCluster <= 0 && bestInc < 0 ) ) // best modularity reached
					break;
								
				// do best merge, set all bestModulNB to bestModulV	
				Set<double[]> s = new HashSet<double[]>();
				for( double[] d : curMap.keySet() )
					if( curMap.get(d) == bestModulNB )
						s.add(d);
				for( double[] d : s ) 
					curMap.put(d, bestModulV);
				
				// update connected modules, 3 steps
				curConMap.get(bestModulV).remove(bestModulNB); 
				
				if( curConMap.containsKey(bestModulNB) )
					curConMap.get(bestModulV).addAll( curConMap.remove(bestModulNB));
				
				for( Set<Integer> set : curConMap.values() ) {
					if( set.contains(bestModulNB ) ) {
						set.remove(bestModulNB);
						set.add(bestModulV);
					}
				}	
			}	
			double curModularity = modularity(graph, curMap);
			
			if( bestMap == null
					|| ( nrCluster <= 0 && curModularity > bestModularity ) // only mod counts
					|| ( nrCluster > 0 && nrCluster - curMap.size() < nrCluster - bestMap.size() ) // closer to desired number of clusters
					|| ( nrCluster > 0 && nrCluster - curMap.size() == nrCluster - bestMap.size() && curModularity > bestModularity ) ) {
				bestModularity = curModularity;				
				bestMap = new HashMap<double[],Integer>(curMap);
			}
		}	
		return bestMap; 
	}
	
	public static Map<double[],Integer> multilevelOptimization( Map<double[],Map<double[],Double>> graph, int rndRestarts ) {
		
		Map<double[],Map<double[],Double>> curGraph = graph;
				
		Map<double[],Set<double[]>> mapping = new HashMap<double[],Set<double[]>>();
		for( double[] d : curGraph.keySet() ) {
			Set<double[]> s = new HashSet<double[]>();
			s.add(d);
			mapping.put(d, s);
		}
				
		while(true) { // no change or modularity does not improve
	
			log.debug("step 1: greedy opt...");
			Map<double[],Integer> modulMap = greedyOptModularity(curGraph, rndRestarts, -1);
			int moduls = new HashSet<Integer>(modulMap.values()).size();
			
			log.debug( modularity(curGraph, modulMap) +", graphsize: "+curGraph.size()+", moduls: "+moduls );
			
			if( curGraph.size() == moduls ) { // greedy opt could not reduce size anymore
				
				Map<double[],Integer> mod = new HashMap<double[],Integer>();
				for( double[] a : modulMap.keySet() ) {
					int c = modulMap.get(a);
					for( double[] b : mapping.get(a) )
						mod.put(b, c);
				}
				return mod;
			}
															
			log.debug("step 2: build new graph...");
			Map<double[],Map<double[],Double>> ng = new HashMap<double[],Map<double[],Double>>();
			
			// build vertices/module representatives
			Map<Integer,double[]> done = new HashMap<Integer,double[]>();
			for( double[] a : curGraph.keySet() ) {
				int c = modulMap.get(a);
				if( done.containsKey(c))
					continue;
				
				done.put(c,a);
				
				Set<double[]> s = new HashSet<double[]>();
				for( double[] b : curGraph.keySet() )
					if( modulMap.get(b) == c )
						s.addAll(mapping.remove(b));
				mapping.put(a, s); // represents now all elements of s 
			}
						
			for( int c : done.keySet() ) {
				double[] v = done.get(c);
				ng.put(v, new HashMap<double[],Double>());
				
				// get connection weights to other modularity
				Map<Integer,Double> wMap = new HashMap<Integer,Double>();
				for( double[] a : curGraph.keySet() ) {
					if( modulMap.get(a) != c )
						continue;
					for( double[] b : curGraph.get(a).keySet() ) {
						int bc = modulMap.get(b);
						double weight = curGraph.get(a).get(b);
						if( wMap.containsKey(bc) )
							wMap.put(bc, wMap.get(bc)+weight);
						else
							wMap.put(bc,weight);
					}
				}
				
				// create connections
				for( int i : wMap.keySet() ) 
					ng.get(v).put(done.get(i), wMap.get(i));
			}
			curGraph = ng;
		}
	}
	
	public static void main(String[] args) {
		UndirectedSparseGraph<double[], double[]> g;
		Reader reader;
		try {
			reader = new FileReader("output/test.graphml");

			Transformer<NodeMetadata, double[]> vtrans = new Transformer<NodeMetadata, double[]>() {
				public double[] transform(NodeMetadata nmd) {
					
					Map<String,String> props = nmd.getProperties();
					List<String> keys = new ArrayList<String>(props.keySet());
					Collections.sort(keys);
					double[] v = new double[keys.size()+1];
					for( int i = 0; i < keys.size(); i++ )
						v[i] = Double.parseDouble( props.get(keys.get(i)) );
					v[v.length-1] = Integer.parseInt( nmd.getId() );
					return v;
				}
			};

			Transformer<EdgeMetadata, double[]> etrans = new Transformer<EdgeMetadata, double[]>() {
				public double[] transform(EdgeMetadata emd) {
					
					Map<String,String> props = emd.getProperties();
					List<String> keys = new ArrayList<String>(props.keySet());
					Collections.sort(keys);
					double[] v = new double[keys.size()];
					for( int i = 0; i < keys.size(); i++ )
						v[i] = Double.parseDouble( props.get(keys.get(i)) ); 
					return v;
				}
			};

			Transformer<HyperEdgeMetadata, double[]> htrans = new Transformer<HyperEdgeMetadata, double[]>() {
				public double[] transform(HyperEdgeMetadata metadata) {
					return new double[]{};
				}
			};

			Transformer<GraphMetadata, UndirectedSparseGraph<double[], double[]>> gtrans = new Transformer<GraphMetadata, UndirectedSparseGraph<double[], double[]>>() {
				public UndirectedSparseGraph<double[], double[]> transform(GraphMetadata gmd) {
					return new UndirectedSparseGraph<double[], double[]>();
				}
			};

			GraphMLReader2<UndirectedSparseGraph<double[], double[]>, double[], double[]> gmlr = new GraphMLReader2<UndirectedSparseGraph<double[], double[]>, double[], double[]>(reader, gtrans, vtrans, etrans, htrans);
			g = gmlr.readGraph();
			log.debug("orig graph: "+g.getVertexCount()+","+g.getEdgeCount());
			
			Map<double[],Map<double[],Double>> graph = new HashMap<double[],Map<double[],Double>>();
			for( double[] v : g.getVertices() ) {
				if( !graph.containsKey(v) )
					graph.put( v, new HashMap<double[],Double>() );
				
				for( double[] nb : g.getNeighbors(v) ) {
					graph.get(v).put(nb, 1.0);
					
					// undirected
					if( !graph.containsKey(nb) ) 
						graph.put( nb, new HashMap<double[],Double>() );
					graph.get(nb).put(v, 1.0);
				}
			}
					
			{
				log.debug("Greedy: ");
				long time = System.currentTimeMillis();
				Map<double[],Integer> map = spawnn.utils.GraphClustering.greedyOptModularity(graph, 10, -1);
				log.debug("took: "+(System.currentTimeMillis()-time)/1000.0);
				Collection<Set<double[]>> ll = spawnn.utils.GraphClustering.modulMapToCluster(map);
				log.debug("communities: "+ll.size());
				log.debug("Modularity: "+spawnn.utils.GraphClustering.modularity(graph, map));
			}
			
			/*{
				log.debug("Multilevel Opt: ");
				Map<double[],Integer> map = spawnn.utils.GraphClustering.multilevelOptimization(graph, 10);
				Collection<Set<double[]>> ll = spawnn.utils.GraphClustering.modulMapToCluster(map);
				log.debug("communities: "+ll.size());
				log.debug("Modularity: "+spawnn.utils.GraphClustering.modularity(graph, map));
			}*/

		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		} catch (GraphIOException e) {
			e.printStackTrace();
		}
	}
}
