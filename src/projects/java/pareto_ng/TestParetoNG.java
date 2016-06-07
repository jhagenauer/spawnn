package pareto_ng;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.Stroke;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import javax.imageio.ImageIO;

import org.apache.commons.collections15.Transformer;
import org.apache.commons.collections15.TransformerUtils;

import edu.uci.ics.jung.algorithms.layout.AbstractLayout;
import edu.uci.ics.jung.algorithms.layout.StaticLayout;
import edu.uci.ics.jung.graph.Graph;
import edu.uci.ics.jung.graph.UndirectedSparseGraph;
import edu.uci.ics.jung.visualization.VisualizationImageServer;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;

public class TestParetoNG {
	public static void main( String[] args ) {
		Random r = new Random();
		List<double[]> samples = new ArrayList<double[]>();
		
		// one f-cluster
		for( int i = 0; i < 200; i++ ) {
			double x = r.nextDouble();
			double y = r.nextDouble();
			double z = 0.5+r.nextDouble()*0.1;
			samples.add( new double[]{x,y,z} );
		}
		
		// one g-cluster
		for( int i = 0; i < 200; i++ ) {
			double x = 0.5+r.nextDouble()*0.1;
			double y = 0.5+r.nextDouble()*0.1;
			double z = r.nextDouble();
			samples.add( new double[]{x,y,z} );
		}
		
		// one f-g-cluster
		/*for( int i = 0; i < 100; i++ ) {
			double x = 0.2+r.nextDouble()*0.1;
			double y = 0.2+r.nextDouble()*0.1;
			double z = 0.2+r.nextDouble()*0.1;
			samples.add( new double[]{x,y,z} );
		}*/
				
		int[] ga = new int[]{0,1};
		Dist<double[]> gDist = new EuclideanDist( ga );
		Dist<double[]> fDist = new EuclideanDist(new int[]{ 2 });
		
		int nrNeurons = 10;
		List<double[]> neurons = new ArrayList<double[]>();
		for (int i = 0; i < nrNeurons; i++) {
			double[] d = samples.get(r.nextInt(samples.size()));
			neurons.add(Arrays.copyOf(d, d.length));
		}
		
		DecayFunction nbRate = new PowerDecay(nrNeurons*2/3, 0.1);
		DecayFunction lrRate = new PowerDecay(0.6, 0.05);
		ParetoNG ng = new ParetoNG(neurons, nbRate, lrRate, gDist, fDist );
		
		int T_MAX = 100000;
		for (int t = 0; t < T_MAX; t++) {
			int j = r.nextInt(samples.size());
			ng.train( (double) t / T_MAX, samples.get(j) );
		}
		
		Set<Set<double[]>> s = new HashSet<>();
		for( double[] x : samples ) {
			List<Set<double[]>> l = ng.getParetoFronts(x);
			s.add( l.get(0) );
			s.add( l.get(1) );
		}
		System.out.println("s: "+s.size());
		System.exit(1);
		
		// build graph
		Map<double[],Map<double[],Integer>> connections = new HashMap<double[],Map<double[],Integer>>();
		for( double[] x : samples ) {
			List<Set<double[]>> fronts = ng.getParetoFronts(x);
			for( double[] a : fronts.get(0) ) {
				double distAa = gDist.dist(a, x);
				double distBa = fDist.dist(a, x);
				for( double[] b : fronts.get(1) ) {					
					double distAb = gDist.dist(b,x);
					double distBb = fDist.dist(b,x);
					
					// a dominates b
					if( (distAa < distAb && distBa <= distBb) || (distAa <= distAb && distBa < distBb) ) {
						
						// just one direction
						if( !connections.containsKey(a) && !connections.containsKey(b) ) {
							connections.put(a, new HashMap<>());
							connections.get(a).put(b, 1);
						} else if( connections.containsKey(a) ) {
							if( !connections.get(a).containsKey(b) ) 
								connections.get(a).put(b, 1 );
							else
								connections.get(a).put(b, connections.get(a).get(b)+1);
						} else if( connections.containsKey(b) ) {
							if( !connections.get(b).containsKey(a) )
								connections.get(b).put(a, 1);
							else
								connections.get(b).put(a, connections.get(b).get(a)+1);
						}
					
					}
				}
			}
			
		}
		
		Map<Integer,Integer> counts = new HashMap<Integer,Integer>();
		for( Map<double[],Integer> m : connections.values() )
			for( int i : m.values() )
				if( !counts.containsKey(i) )
					counts.put(i, 1);
				else
					counts.put(i, counts.get(i)+1);
		System.out.println(counts);
							
		// to jung-graph, no costs ATM
		Graph<double[], double[]> g = new UndirectedSparseGraph<double[], double[]>();
		for( double[] a : connections.keySet() )
			for( double[] b : connections.get(a).keySet() ) {

			if (!g.getVertices().contains(a))
				g.addVertex(a);
			if (!g.getVertices().contains(b))
				g.addVertex(b);

			g.addEdge(new double[] { connections.get(a).get(b) }, a, b);
		}
		
		// draw jung graph

		//AbstractLayout<double[],double[]> al = new KKLayout<double[], double[]>(g);
		int border = 12; // TODO this can be done more elegant
		Dimension dim = new Dimension(1000, 1000);
		double minX = Double.POSITIVE_INFINITY, minY = Double.POSITIVE_INFINITY, maxX = Double.NEGATIVE_INFINITY, maxY = Double.NEGATIVE_INFINITY;
		for (double[] d : g.getVertices()) {
			minX = Math.min(minX, d[ga[0]]);
			maxX = Math.max(maxX, d[ga[0]]);
			minY = Math.min(minY, -d[ga[1]]);
			maxY = Math.max(maxY, -d[ga[1]]);
		}
		Map<double[], Point2D> map = new HashMap<double[], Point2D>();
		for (double[] d : g.getVertices()) {
			// keep aspect ratio
			double s1 = Math.max(maxX - minX, maxY - minY);
			double s2 = Math.max(dim.getWidth(), dim.getHeight());
			map.put(d, new Point2D.Double((s2 - 2 * border) * (d[ga[0]] - minX) / s1 + border, (s2 - 2 * border) * (-d[ga[1]] - minY) / s1 + border));
		}
		Transformer<double[], Point2D> vertexLocations = TransformerUtils.mapTransformer(map);
		AbstractLayout<double[],double[]> al = new StaticLayout<double[], double[]>(g, vertexLocations);
		
		
		VisualizationImageServer<double[], double[]> vis = new VisualizationImageServer<double[], double[]>(al,new Dimension(1000, 1000));
		vis.setBackground(Color.WHITE);
		
		/*vis.getRenderContext().setVertexFillPaintTransformer(vv.getRenderContext().getVertexFillPaintTransformer());
		vis.getRenderContext().setVertexStrokeTransformer( vv.getRenderContext().getVertexStrokeTransformer());
		vis.getRenderContext().setVertexDrawPaintTransformer( vv.getRenderContext().getVertexDrawPaintTransformer());*/
				
		//vis.getRenderContext().setEdgeShapeTransformer(vv.getRenderContext().getEdgeShapeTransformer());
		vis.getRenderContext().setEdgeStrokeTransformer(new Transformer<double[], Stroke>() {
			@Override
			public Stroke transform(double[] edge) {
				return new BasicStroke( (float)edge[0] );
			}});
		
		double w = al.getSize().getWidth();
		double h = al.getSize().getHeight();
		Point2D.Double center = new Point2D.Double(w / 2, h / 2);
						
		BufferedImage bufImage = (BufferedImage) vis.getImage(center,new Dimension(al.getSize()));
				
		try {		
			FileOutputStream stream = new FileOutputStream("output/jungGraph.png");		
			Graphics2D gr = bufImage.createGraphics();
			gr.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			ImageIO.write(bufImage, "PNG", stream);
			stream.flush();
			stream.close();
		} catch( Exception e ) {
			
		}
	}
}
