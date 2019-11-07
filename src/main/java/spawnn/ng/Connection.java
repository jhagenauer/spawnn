package spawnn.ng;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamWriter;

// undirected
public class Connection {
	private double[] a,b;
	
	public Connection(double[] a, double[] b ) {
		this.a = a;
		this.b = b;
	}
	
	public double[] getA() {
		return a;
	}
	
	public double[] getB() {
		return b;
	}

	@Override
	public int hashCode() {
		return 31 + a.hashCode() + b.hashCode();
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null) 
			return false;
		if (getClass() != obj.getClass()) 
			return false;
		
		Connection other = (Connection) obj;	
		if( !(a == other.a && b == other.b ) && !( a == other.b && b == other.a ) )
			return false;
		return true;
	}

	@Override
	public String toString() {
		return "Connection [a=" + Arrays.toString(a) + ", b=" + Arrays.toString(b) + "]";
	}
	
	public static Set<double[]> getNeighbors( Collection<Connection> cons, double[] d, int depth ) {
		Set<double[]> l = new HashSet<double[]>();
		if( depth == 0 )
			return l;
						
		for( Connection c : cons ) {
			double[] a = c.getA();
			double[] b = c.getB();
			
			double[] nb = null;
			if( d == a ) 
				nb = b;
			else if( d == b )
				nb = a;
			else 
				continue;
						
			if( !l.contains(nb ) ) {
				l.add( nb );
				l.addAll( getNeighbors( cons, nb, depth-1));
			}
		}
		
		return l;
	}
			
	public static int dist( Collection<Connection> cons, double[] from, double[] to ) {		
		Set<double[]> openList = new HashSet<double[]>();
		Map<double[],Integer> distMap = new HashMap<double[],Integer>();
		
		openList.add( from );
		distMap.put( from, 0 );
		
		while( !openList.isEmpty() ) {
			
			// find closest from distMap 
			int min = Integer.MAX_VALUE;
			double[] curNode = null;
			for( double[] v : openList ) {
				if( distMap.get(v) < min ) {
					min = distMap.get(v);
					curNode = v;
				}
			}
			openList.remove(curNode);
						
			if( curNode == to ) // path found, reconstruct path
				return distMap.get(curNode);
															
			Set<double[]> arcs = getNeighbors( cons, curNode, 1);
			if( arcs == null ) 
				continue;
			
			for( double[] succesor : arcs ) {
				int d = min + 1;
				
				if( !distMap.containsKey(succesor) || d < distMap.get(succesor) ) {
					distMap.put( succesor, d );
					openList.add( succesor );
				} 
			}
		}
		// no connection present
		return Integer.MAX_VALUE;
	}
		
	// iGraph works better with GML.. does it?
	public static void saveAsGML( Collection<double[]> nodes, Collection<Connection> cons, OutputStream os ) {
		List<double[]> ns = new ArrayList<double[]>(nodes);
		StringBuffer sb = new StringBuffer();
		sb.append("graph [\n");
		sb.append("directed 0\n");
		for( double[] d : ns ) {
			sb.append("node [\n");
			sb.append("id "+ns.indexOf(d)+"\n");
			//sb.append("label \""+d.hashCode()+"\"\n");
			sb.append("]\n");
		}
		for( Connection c : cons ) {
			sb.append("edge [\n");
			sb.append("source "+ns.indexOf(c.getA() )+"\n");
			sb.append("target "+ns.indexOf(c.getB() )+"\n");
			//sb.append("label \""+c.hashCode()+"\"\n");
			sb.append("]\n");
		}
		sb.append("]\n");
		OutputStreamWriter osw = null; 
		try {
			osw = new OutputStreamWriter(os); 
			osw.write(sb.toString());
			osw.flush();
		} catch (IOException e) {
			
			e.printStackTrace();
		} finally {
			try {
				osw.close();
			} catch (IOException e1) {
				e1.printStackTrace();
			}
		}	
	}	
		
	public static void writeGraphML(Map<Connection, Integer> conns, String fn) {
		Set<double[]> used = new HashSet<double[]>();
		for (Connection c : conns.keySet()) {
			used.add(c.getA());
			used.add(c.getB());
		}

		List<double[]> vertices = new ArrayList<double[]>(used);
		List<Connection> edges = new ArrayList<Connection>(conns.keySet());

		File graphmlFile = new File(fn);
		try {
			OutputStream out = new FileOutputStream(graphmlFile);

			XMLOutputFactory factory = XMLOutputFactory.newInstance();
			XMLStreamWriter writer = factory.createXMLStreamWriter(out);

			writer.writeStartDocument();
			writer.writeStartElement("graphml");

			// node attributes
			for (int i = 0; i < vertices.get(0).length; i++) {
				writer.writeStartElement("key");
				writer.writeAttribute("id", "n" + i);
				writer.writeAttribute("for", "node");
				writer.writeAttribute("attr.name", "n" + i);
				writer.writeAttribute("attr.type", "double");
				writer.writeEndElement();
			}

			// edge attributes
			writer.writeStartElement("key");
			writer.writeAttribute("id", "e0");
			writer.writeAttribute("for", "edge");
			writer.writeAttribute("attr.name", "Weight");
			writer.writeAttribute("attr.type", "long");
			writer.writeEndElement();

			writer.writeStartElement("graph");
			writer.writeAttribute("edgedefault", "undirected");

			// write nodes
			for (double[] v : vertices) {

				writer.writeStartElement("node");
				writer.writeAttribute("id", "" + vertices.indexOf(v));

				for (int i = 0; i < v.length; i++) {
					writer.writeStartElement("data");
					writer.writeAttribute("key", "n" + i);
					writer.writeCharacters(v[i] + "");
					writer.writeEndElement(); // end data
				}

				writer.writeEndElement(); // end node
			}

			// write edges
			for (Connection e : edges) {

				writer.writeStartElement("edge");
				writer.writeAttribute("id", "" + edges.indexOf(e));
				writer.writeAttribute("source", vertices.indexOf(e.getA()) + "");
				writer.writeAttribute("target", vertices.indexOf(e.getB()) + "");

				writer.writeStartElement("data");
				writer.writeAttribute("key", "e0");
				writer.writeCharacters("" + conns.get(e));
				writer.writeEndElement(); // end data

				writer.writeEndElement(); // end edge
			}

			writer.writeEndElement(); // end graph
			writer.writeEndElement(); // end graphml
			writer.writeEndDocument();
			writer.flush();
			writer.close();
			out.flush();
			out.close();
		} catch (XMLStreamException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
