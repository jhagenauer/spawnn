package spawnn.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamWriter;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;

import spawnn.dist.Dist;

public class RegionUtils {
	
	private static Logger log = Logger.getLogger(RegionUtils.class);
		
	// cm = contiguitiy map, nodes = list of nodes to build distance matrix... order matters!!!
	public static int[][] getDistMatrix( Map<double[],Set<double[]>> cm, List<double[]> nodes ) {
		int dist[][] = new int[nodes.size()][nodes.size()];
						
		for( int i = 0; i < dist.length; i++ ) {
			for( int j = i; j < dist.length; j++ ) {				
				if( i == j )
					dist[i][j] = 0;
				else { 
					double[] a = nodes.get(i);
					double[] b = nodes.get(j);
					
					if( cm.get( a ).contains( b ) )
						dist[i][j] = 1;
					else
						dist[i][j] = 99999;
				}
			}
		}
						
		log.debug("building ASP...");
		for( int k = 0; k < dist.length; k++ ) {
			for( int i = 0; i < dist.length; i++ ) {		
				for( int j = i; j < dist.length; j++ ) {
											
					if( k < i ) {
						if( dist[k][j] + dist[k][i] < dist[i][j] ) {
							dist[i][j] = dist[k][j] + dist[k][i];
						}			
					} else if( k < j ) { // i <= k
						if( dist[i][k] + dist[k][j] < dist[i][j] ) {
							dist[i][j] = dist[i][k] + dist[k][j];
						}
					} else if( k > j ) { 
						if( dist[i][k] + dist[j][k] < dist[i][j] ) {
							dist[i][j] = dist[i][k] + dist[j][k];
						}
					}
					
				}		
			}
		}
		log.debug("done.");
		return dist;
	}
	
	public static boolean isContiugous( Map<double[], Set<double[]>> cm, Set<double[]> cluster ) {
		if( cluster.isEmpty() )
			return true;
		
		Set<double[]> visited = getContiugousSubcluster( cm, cluster, cluster.iterator().next() );					
		if( visited.size() != cluster.size() ) 
			return false;
		else return true;
	}
	
	public static Set<Set<double[]>> getAllContiguousSubcluster( Map<double[],Set<double[]>> cm, Set<double[]> cluster ) {
		Set<Set<double[]>> all = new HashSet<Set<double[]>>();
		Set<double[]> ds = new HashSet<double[]>(cluster);
		
		while( !ds.isEmpty() ) {
			double[] d = ds.iterator().next(); 
			Set<double[]> sub = getContiugousSubcluster(cm, cluster, d); 	
			all.add(sub);
			ds.removeAll(sub);
		}
		return all;
	}
	
	public static Set<double[]> getContiugousSubcluster( Map<double[],Set<double[]>> cm, Set<double[]> cluster, double[] d ) {
					
		Set<double[]> visited = new HashSet<double[]>();
		List<double[]> openList = new ArrayList<double[]>();
		openList.add( d );
			
		while( !openList.isEmpty() ) {
			double[] cur = openList.remove(openList.size()-1);
			visited.add(cur);
				
			// get neighbors not visited and add to openList
			for( double[] nb : cm.get(cur) )
				if( !visited.contains( nb ) && cluster.contains(nb) )
					openList.add( nb );
		}
		
		return visited;
	}
			
	public static double getHeterogenity( Collection<Set<double[]>> clusters, int[] fa ) {
		double sum = 0;
		for( Set<double[]> cluster : clusters  ) 
			sum += getSSD( cluster, fa);
		return sum;
	}
	
	public static double getSSD( Set<double[]> cluster, int[] fa ) {
		double sum = 0;
		int cs = cluster.size();
		
		if( cs == 0 )
			return sum;
			
		double[] mean = new double[fa.length];
		for( double[] d : cluster )
			for( int j = 0; j < fa.length; j++ )
				mean[j] += d[fa[j]]/cs;
				
		for( double[] d : cluster ) {
			double s = 0;
			for( int j = 0; j < fa.length; j++ )
				s += Math.pow( d[fa[j]] - mean[j], 2 );
			
			sum += s/fa.length; // NOTE: Guo introduced the division, Assuancao does not do this!
		}

		return sum;
	}
		
	public static Map<double[], Set<double[]>> deriveQueenContiguitiyMap( List<double[]> samples, List<Geometry> geoms ) {
		Map<double[], Set<double[]>> cm = new HashMap<double[],Set<double[]>>();
		
		for( int i = 0; i < samples.size(); i++ ) {
			double[] a = samples.get(i);
			Geometry ag = geoms.get(i);
			
			cm.put( a, new HashSet<double[]>() );
			
			for( int j = 0; j < samples.size(); j++ ) {
				double[] b = samples.get(j);
				Geometry bg = geoms.get(j);
				
				if( bg.touches(ag) || bg.intersects(ag) )
					cm.get(a).add(b);				
			}
		}
		return cm;
	}
	
	public static Map<double[], Set<double[]>> readContiguitiyMap( List<double[]> samples, String fn ) {
		Map<double[], Set<double[]>> cm = new HashMap<double[], Set<double[]>>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(fn));
			String line = br.readLine();
			while ((line = br.readLine()) != null) {
				String[] split = line.split(",");
				int a = Integer.parseInt(split[0]);
				int b = Integer.parseInt(split[1]);

				double[] da = samples.get(a), db = samples.get(b);

				if (!cm.containsKey(da))
					cm.put(da, new HashSet<double[]>());

				cm.get(da).add(db);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return cm;		
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
	
	public static Collection<Set<double[]>> readRedcapResults( List<double[]> samples, int regions,String fn ) {
		Map<Integer,Set<double[]>> map = new HashMap<Integer,Set<double[]>>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(fn));
			String line = br.readLine(); // ignore header
			
			while ((line = br.readLine()) != null) {
				String[] split = line.split(",");
				int num = Integer.parseInt(split[0]);
				
				int region = Integer.parseInt(split[regions]);
				
				if( !map.containsKey(region) )
					map.put( region, new HashSet<double[]>() );
			
				map.get(region).add( samples.get(num) );
		
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return map.values();
	}
	
	// Note: supp. igraph algorithms spinnglas, label.propagation.community, fastgreedy, walktrap
	public static void writeAsGraphML( Map<double[], Set<double[]>> cm, int idCol, Dist<double[]> d, OutputStream os) {
				
		Set<double[]> allNodes = new HashSet<double[]>(cm.keySet());
		for( Set<double[]> s : cm.values() )
			allNodes.addAll(s);
		
		try {
			XMLOutputFactory factory = XMLOutputFactory.newInstance();
			XMLStreamWriter writer = factory.createXMLStreamWriter(os);
						
			writer.writeStartDocument();
			
			writer.writeStartElement("graphml");
			writer.writeDefaultNamespace("http://graphml.graphdrawing.org/xmlns");
			
			// node attributes
			for (int i = 0; i < allNodes.iterator().next().length; i++) {
				writer.writeStartElement("key");
				writer.writeAttribute("id", "n" + i);
				writer.writeAttribute("for", "node");
				writer.writeAttribute("attr.name", "n" + i);
				writer.writeAttribute("attr.type", "double");
				writer.writeEndElement();
			}
						
			writer.writeStartElement("key");
			writer.writeAttribute("id", "weight");
			writer.writeAttribute("for", "edge");
			writer.writeAttribute("attr.name", "weight");
			writer.writeAttribute("attr.type", "double");
			writer.writeEndElement();
			
			writer.writeStartElement("graph");
			writer.writeAttribute("id", "G");
			writer.writeAttribute("edgedefault", "undirected");
			
			Map<double[],Integer> ids = new HashMap<double[],Integer>();
			int j = 0;
			for( double[] n : allNodes ) {
				if( idCol >= 0 )
					ids.put(n, (int)n[idCol]);
				else
					ids.put(n,j++);
					
			}
						
			for( double[] n : allNodes ) {
				writer.writeStartElement("vertex");
				writer.writeAttribute("id", ids.get(n)+"" );// id column
				
				for (int i = 0; i < n.length; i++) {
					writer.writeStartElement("data");
					writer.writeAttribute("key", "n" + i);
					writer.writeCharacters(n[i] + "");
					writer.writeEndElement(); // end data
				}
				
				writer.writeEndElement(); // end node
			}
								
			int i = 0;
			for( double[] a : cm.keySet() ) {
				for( double[] b : cm.get(a)) {
					
					// skip self-references
					if( a == b )
						continue;
										
					writer.writeStartElement("edge");
					writer.writeAttribute("id", i+++"");
					writer.writeAttribute("source", ids.get(a)+"" );
					writer.writeAttribute("target", ids.get(b)+"" );
										
					writer.writeStartElement("data");
					writer.writeAttribute("key","weight");
					writer.writeCharacters(d.dist(a, b)+"");
					writer.writeEndElement(); // end data
					
					writer.writeEndElement(); // end edge
				}
			}
			
			writer.writeEndElement(); // end graph
			writer.writeEndElement(); // end graphml
			writer.writeEndDocument();
			writer.flush();
			writer.close();
			os.flush();
			os.close();
		} catch( XMLStreamException e ) {
			e.printStackTrace();
		} catch( IOException e ) {
			e.printStackTrace();
		}	
	}
}
