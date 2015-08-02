package clustering_cng;


import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamWriter;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.Connection;
import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class MultiCNG_Diamond {

	private static Logger log = Logger.getLogger(MultiCNG_Diamond.class);
	
	public static void main(String[] args) {

		final Random r = new Random();
		final int T_MAX = 100000;

		SpatialDataFrame sd = DataUtils.readShapedata( new File("data/diamond/diamond.shp") , new int[] {}, true);
		final List<double[]> samples = sd.samples;
		final String[] header = sd.names.toArray(new String[]{} );
		
		for( int i = 0; i < header.length; i++ )
			header[i] = header[i].toLowerCase();
		
		final int[] ga = new int[] { 0, 1 };
		final int[] fa = { 2 };
		
		final Dist<double[]> gDist = new EuclideanDist( ga);
		final Dist<double[]> fDist = new EuclideanDist( fa);
		
		ExecutorService es = Executors.newFixedThreadPool(4);
		
		for( int run = 0; run < 100; run++ ) {
			for( int k = 1; k<=25; k++ ) {
				
				final int RUN = run, K = k;
				
				es.execute( new Runnable() {
					
					@Override
					public void run() {
						// train ng
						Sorter<double[]> bmuGetter = new KangasSorter<double[]>( gDist, fDist, K );
						NG ng = new NG(25, 10, 0.01, 0.5, 0.005, samples.get(0).length, bmuGetter );
								
						for (int t = 0; t < T_MAX; t++) {
							double[] x = samples.get(r.nextInt(samples.size() ) );
							ng.train( (double)t/T_MAX, x );
						}
						
						Map<double[],String> label = new HashMap<double[],String>();
						for( double[] d : ng.getNeurons() ) 
							label.put(d, "n"+ng.getNeurons().indexOf(d) );
												
						// train chl
						Map<Connection,Integer> conns = new HashMap<Connection,Integer>();
						for( double[] x : samples ) {
							bmuGetter.sort(x, ng.getNeurons() );
							List<double[]> bmuList = ng.getNeurons();
											
							Connection c = new Connection( bmuList.get(0), bmuList.get(1) );
							if( !conns.containsKey(c) )
								conns.put( c, 1);
							else
								conns.put( c, conns.get(c) + 1 );
						}
						
						writeGraphML(conns, label, header, "output/"+RUN+"_"+K+".graphml");
						writeCSV( NGUtils.getBmuMapping(samples, ng.getNeurons(), bmuGetter ), label, header, "output/"+RUN+"_"+K+".csv" );
						
					}
				});
			}
		}
		es.shutdown();
	}
	
	public static void writeGraphML(Map<Connection, Integer> conns, Map<double[], String> label, String[] header, String fn) {
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
				writer.writeAttribute("attr.name", header[i] );
				writer.writeAttribute("attr.type", "double");
				writer.writeEndElement();
			}

			// edge attributes
			writer.writeStartElement("key");
			writer.writeAttribute("id", "e0");
			writer.writeAttribute("for", "edge");
			writer.writeAttribute("attr.name", "weight");
			writer.writeAttribute("attr.type", "long");
			writer.writeEndElement();

			writer.writeStartElement("graph");
			writer.writeAttribute("edgedefault", "undirected");

			// write nodes
			for (double[] v : vertices) {

				writer.writeStartElement("node");
				writer.writeAttribute("id", "" + label.get(v) );

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
				writer.writeAttribute("source", label.get(e.getA()) + "");
				writer.writeAttribute("target", label.get(e.getB()) + "");

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
	
	public static void writeCSV( Map<double[], Set<double[]>> bmuMapping, Map<double[], String> label, String[] header, String fn) {
		BufferedWriter w = null;
		
		try {
			w = new BufferedWriter(new FileWriter(fn));
			
			w.write("neuron,");
			for( int i = 0; i < header.length; i++ ) {
				w.write(header[i]);
				if( i < header.length-1 )
					w.write(",");
			}
			w.write("\n");
			
			for( double[] n : bmuMapping.keySet() ) {
				for( double[] x : bmuMapping.get(n) ) {
					StringBuffer sb = new StringBuffer();
					sb.append(label.get(n)+",");
					for( int i = 0; i < x.length; i++ ) {
						sb.append(x[i]);
						if( i < x.length-1 )
							sb.append(",");
					}
					w.write(sb.toString()+"\n");
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				w.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
}
