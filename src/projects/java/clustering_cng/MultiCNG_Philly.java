package clustering_cng;


import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
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
import org.geotools.geometry.jts.JTS;
import org.geotools.referencing.CRS;
import org.opengis.referencing.crs.CoordinateReferenceSystem;
import org.opengis.referencing.operation.MathTransform;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.Point;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.Connection;
import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class MultiCNG_Philly {

	private static Logger log = Logger.getLogger(MultiCNG_Philly.class);
	
	public static void main(String[] args) {
		
		File file = new File("data/philadelphia/tracts/philadelphia_tracts_with_pop.shp");
		final SpatialDataFrame sd = DataUtils.readShapedata(file, new int[] {}, false);
		
		List<double[]> samplesOrig = sd.samples;
		List<Geometry> geoms = new ArrayList<Geometry>();
		
		CoordinateReferenceSystem targetCRS = null;
		try {			
			CoordinateReferenceSystem sourceCRS = sd.crs;
			targetCRS =  CRS.decode("EPSG:2272");
			MathTransform transform = CRS.findMathTransform(sourceCRS, targetCRS, true);
		
			// ugly, but easy
			for (int i = 0; i < samplesOrig.size(); i++) {
				double[] d = samplesOrig.get(i);
				
				Geometry g = JTS.transform(sd.geoms.get(i), transform);			
				Point p = g.getCentroid();
				d[0] = p.getX();
				d[1] = p.getY();
				
				geoms.add(g);
			}
			
		} catch( Exception e ) {
			e.printStackTrace();
		}
		

		final List<double[]> samples = new ArrayList<double[]>();
		for (double[] d : samplesOrig) {

			double[] nd = new double[] { 
					samples.size(), // id
					d[0], // x
					d[1], // y

					d[2], // pop

					// ---- age ----
					(d[3] + d[4] + d[5] + d[6] + d[7]) / d[2], // age 0 to 24
					(d[8] + d[9] + d[10] + d[11] + d[12] + d[13] + d[14] + d[15]) / d[2], // age 25 to 64
					(d[16] + d[17] + d[18] + d[19] + d[20]) / d[2], // age 65 and older
					// d[59], // median age

					// ---- race ----
					d[79] / d[2], // white
					d[80] / d[2], // black
					// d[81], // indian
					d[82] / d[2], // asian
					d[115] / d[2], // hispanic
					// d[124], // white, not hispanic
					// d[125], // black, not hispanic

					// ---- households ----
					// d[151], // households
					// d[152], // family households
					d[168], // avg household size
					// d[169], // avg family size

					// ---- housing ----
					// d[170], // total housing units
					// d[171], // occupied
					// d[172], // vacant
					d[171] / d[170], // occupied-rate

					// d[181], // total occupied housing units
					// d[182], // owner-occupied housing units
					d[183] / d[170], // renter-occupied housing units
			};

			// "repair" NANs
			for (int i = 0; i < nd.length; i++)
				if (Double.isNaN(nd[i]))
					nd[i] = 0;

			samples.add(nd);
		}
		
		final String header[] = new String[] { "id2", "x", "y", "pop", "0to24", "25to64", "65older", "white", "black", "asian", "hispanic", "avgHHSize", "occup", "rentOccup" };

		int[] fa = new int[samples.get(0).length - 3];
		for (int i = 3; i < fa.length; i++)
			fa[i] = i;

		int[] ga = { 1, 2 };
		
		DataUtils.writeShape(samples, geoms, header, targetCRS, "output/data_philly.shp");
		System.exit(1);
				
		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);

		final List<double[]> pristineSamples = new ArrayList<double[]>();
		for( double[] d : samples )
			pristineSamples.add( Arrays.copyOf(d, d.length));
		
		DataUtils.zScoreColumns(samples, fa);

		final Random r = new Random();
		final int T_MAX = 100000;

		ExecutorService es = Executors.newFixedThreadPool(16);
		
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
						
						List<double[]> neuronList = new ArrayList<double[]>(ng.getNeurons()); // fixes order for neuron-ids
						writeGraphML(conns, neuronList, header, "output/"+RUN+"_"+K+".graphml");
												
						Map<double[],Set<double[]>> mapping = NGUtils.getBmuMapping(samples, ng.getNeurons(), bmuGetter );
						
						// shape stores non-transformed clusters!
						List<double[]> ns = new ArrayList<double[]>();
						for( int i = 0; i < samples.size(); i++ ) {
							double[] d = samples.get(i);
							
							int n = -1;
							for( double[] neuron : mapping.keySet() )
								if( mapping.get(neuron).contains(d) )
									n = neuronList.indexOf(neuron);		
							
							double[] dOrig = pristineSamples.get(i); // untransformed d
							double[] nd = Arrays.copyOf(dOrig, dOrig.length+1);
							nd[nd.length-1] = n;
							
							ns.add(nd);
						}
												
						String[] nh  = Arrays.copyOf(header, header.length+1);
						nh[nh.length-1] = "neuron";
						
						DataUtils.writeShape(ns, sd.geoms, nh, sd.crs, "output/"+RUN+"_"+K+".shp"); //
					}
				});
			}
		}
		es.shutdown();
	}
	
	public static void writeGraphML(Map<Connection, Integer> conns, List<double[]> nn, String[] header, String fn) {
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
				writer.writeAttribute("id", "" + nn.indexOf(v) );

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
				writer.writeAttribute("source", nn.indexOf(e.getA()) + "");
				writer.writeAttribute("target", nn.indexOf(e.getB()) + "");

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
