package context.cng;


import static spawnn.ng.Connection.writeGraphML;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.jdom.Document;
import org.jdom.Element;
import org.jdom.Namespace;
import org.jdom.input.SAXBuilder;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.Connection;
import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;

import com.vividsolutions.jts.geom.Geometry;

// Topology representing network
public class NeuralGasCHL {
	
	public static void main( String args[] ) {
		Random r = new Random();
				
		int T_MAX = 200000;
		
		List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("output/rregions.shp"), new int[]{}, true);
		List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(new File("output/rregions.shp"));
				
		//List<double[]> samples = DataUtil.loadSamplesFromShapeFile(new File("output/ndiffdenssquares.shp"), new int[]{}, true);
		//List<Geometry> geoms = DataUtil.loadGeometriesFromShapeFile(new File("output/ndiffdenssquares.shp"));

		Map<Integer,Set<double[]>> cl = new HashMap<Integer,Set<double[]>>();
		for( double[] d : samples ) {
			int c = (int)d[3];
			
			if( !cl.containsKey(c) )
				cl.put(c, new HashSet<double[]>() );
			
			cl.get(c).add(d);
		}
				
		int[] fa = new int[]{ 2 };
		int[] ga = new int[]{ 0, 1 };
		
		Dist eDist = new EuclideanDist();
		Dist gDist = new EuclideanDist(ga);
		Dist fDist = new EuclideanDist(fa);
				
		int numNeurons = 25;
		
		Map<Connection,Integer> cons = new HashMap<Connection,Integer>();
		Sorter bmuGetter = new KangasSorter(gDist, fDist, 3 ); 
				
		double tInit = 20, tFinal = 200;
		NG ng = new NG(numNeurons, (double)numNeurons/2, 0.01, 0.5, 0.005, samples.get(0).length, bmuGetter );
		for (int t = 0; t < T_MAX; t++) {
			double[] x = samples.get(r.nextInt( samples.size() ) );
			ng.train( (double)t/T_MAX, x );
			
			bmuGetter.sort(x, ng.getNeurons() );
			cons.put( new Connection( ng.getNeurons().get(0), ng.getNeurons().get(1) ),	0);
					
			// increase age of all cons
			for( Connection c : cons.keySet() ) 
				cons.put( c, cons.get(c)+1 );
					
			double tA = tInit*Math.pow( tFinal/tInit, (double)t/T_MAX );
			
			List<Connection> drop = new ArrayList<Connection>();
	 		for( Connection c : cons.keySet() )
				if( cons.get(c) > tA )
					drop.add(c);
	 	 		
	 		cons.keySet().removeAll(drop);
		}
		
		Map<double[],Set<double[]>> cluster = new HashMap<double[],Set<double[]>>();
		for( double[] w : ng.getNeurons() )
			cluster.put( w, new HashSet<double[]>() );
		for( double[] d : samples ) {
			bmuGetter.sort(d, ng.getNeurons() );
			double[] bmu = ng.getNeurons().get(0);
			cluster.get(bmu).add(d);
		}
						
		System.out.println("Connections: "+cons.size() );
		
		System.out.println("cNG clustering results: ");
		System.out.println("cluster: "+cluster.size());
		System.out.println("NMI: "+DataUtils.getNormalizedMutualInformation(cluster.values(), cl.values()));
		
		// get used or connected ones
		Set<double[]> usedNeurons = new HashSet<double[]>();
		for( double[] d : cluster.keySet() )
			if( !cluster.get(d).isEmpty() )
				usedNeurons.add(d);
		for( Connection c : cons.keySet() ) {
			usedNeurons.add( c.getA() );
			usedNeurons.add( c.getB() );
		}
				
		try {
			Drawer.geoDrawCluster(cl.values(), samples, geoms, new FileOutputStream("output/ngchl_cluster_orig.png"), false);
			Drawer.geoDrawCluster(cluster.values(), samples, geoms, new FileOutputStream("output/ngchl_cluster.png"), false);
						
			Connection.writeGraphML(cons, "output/ngchl.xml" );
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		/*Graph<double[], String> g = new UndirectedSparseGraph<double[], String>();
		for( Connection c : cons.keySet() ) {
			if( !g.getVertices().contains(c.getA() ) )
				g.addVertex(c.getA());
			if( !g.getVertices().contains(c.getB() ) )
				g.addVertex(c.getB());
			g.addEdge( g.getEdges().size()+"", c.getA(), c.getB() );
		}
		
		FRLayout2<double[],String> layout = new FRLayout2<double[], String>(g);
		VisualizationImageServer<double[], String> srv = new VisualizationImageServer<double[], String>(layout, new Dimension(1200, 1200));
		BufferedImage bufImg = (BufferedImage)srv.getImage( new Point(0,0),  new Dimension(1200, 1200));
		
		try {
			ImageIO.write(bufImg, "png", new FileOutputStream("output/ngchl_jung_graph.png") );
		} catch (IOException ex) {
			ex.printStackTrace();
		}*/
		
		System.out.println("run script for graphclustering, than press return...");
		try {
			System.in.read();
		} catch (IOException e) {
			e.printStackTrace();
		}
				
		// build graph clustering from communities, use JDOM for convenience
		Map<Integer,Set<double[]>> graphCluster = new HashMap<Integer,Set<double[]>>();
		  try {
			  SAXBuilder builder = new SAXBuilder();
              Document doc = builder.build( new File("output/communities.xml") );
              Element root = doc.getRootElement();
              Namespace ns = root.getNamespace();
                          
              for(Object o : root.getChild("graph",ns).getChildren("node",ns) ) {
            	  Element node = (Element)o;
            	  int membership = -1;
            	  double[] v = new double[samples.get(0).length];
            	  for( Object o2 : node.getChildren("data",ns) ) {
            		  Element data = (Element)o2;
            		  String keyValue = data.getAttributeValue("key");
            		  if( keyValue.equals("membership") ) 
            			  membership = Integer.parseInt( data.getText() );
            		  else if( keyValue.substring(0, 2).equals("v_") ) {
            			  int i = Integer.parseInt( keyValue.substring(2) );
            			  v[i] = Double.parseDouble( data.getText() );
            		  }
            	  }		
            	  if( !graphCluster.containsKey(membership) && membership >= 0 )
            		  graphCluster.put( membership, new HashSet<double[]>() );
            	  
            	  double[] nearest = null;
            	  double bestDist = Double.MAX_VALUE;
            	  for( double[] d : usedNeurons ) {
            		  if( nearest == null || eDist.dist(d, v) < bestDist ) {
            			  nearest = d;
            			  bestDist = eDist.dist(d,v);
            		  }
            	  }
            	  graphCluster.get(membership).add(nearest);
              }
         } catch( Exception ex ) {
              ex.printStackTrace();
         }
         		
		// reconstruct sample clustering from graph-clustering  
        List<Set<double[]>> cluster2 = new ArrayList<Set<double[]>>();
        for( Set<double[]> s : graphCluster.values() ) {
        	Set<double[]> mapped = new HashSet<double[]>();
        	for( double[] d : s )
        		mapped.addAll( cluster.get(d));
        	cluster2.add(mapped);
        }
        
        System.out.println("Graph clustering results: ");
		System.out.println("cluster: "+cluster2.size());
		System.out.println("NMI: "+DataUtils.getNormalizedMutualInformation(cluster2, cl.values()));
        
        try {
			Drawer.geoDrawCluster(cluster2, samples, geoms, new FileOutputStream("output/ngchl_cluster2.png"), false);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
