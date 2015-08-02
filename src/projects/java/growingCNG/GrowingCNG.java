package growingCNG;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import javax.imageio.ImageIO;

import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.geotools.geometry.jts.ReferencedEnvelope;
import org.geotools.map.FeatureLayer;
import org.geotools.map.Layer;
import org.geotools.map.MapContent;
import org.geotools.renderer.GTRenderer;
import org.geotools.renderer.lite.StreamingRenderer;
import org.geotools.styling.Mark;
import org.geotools.styling.SLD;
import org.geotools.styling.StyleBuilder;
import org.opengis.referencing.crs.CoordinateReferenceSystem;

import spawnn.dist.Dist;
import spawnn.ng.Connection;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.utils.NGUtils;
import spawnn.som.grid.GridPos;
import spawnn.utils.DataUtils;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryCollection;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.LineString;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.triangulate.VoronoiDiagramBuilder;

public class GrowingCNG {
	
	protected List<double[]> neurons = null;
	protected double lrB, lrN, alpha, beta, ratio;
	protected Map<Connection,Integer> cons;
	protected Dist<double[]> distA, distB;
	protected Map<double[],Double> errors;
	protected int aMax, lambda;
	protected Random r = new Random();
	protected Sorter<double[]> sorter;
		
	public GrowingCNG( Collection<double[]> neurons, double lrB, double lrN, Dist<double[]> distA, Dist<double[]> distB, double ratio, int aMax, int lambda, double alpha, double beta ) {
		this.lrB = lrB;
		this.lrN = lrN;
		this.distA = distA;
		this.distB = distB;
		this.ratio = ratio;
		this.cons = new HashMap<Connection,Integer>();
		this.aMax = aMax;
		this.lambda = lambda;
		this.alpha = alpha;
		this.beta = beta;
		this.neurons = new ArrayList<double[]>(neurons);
		
		this.errors = new HashMap<double[],Double>();
		for( double[] n : this.neurons )
			this.errors.put( n, 0.0 );
		
		this.sorter = new DefaultSorter<double[]>(distB);
	}
	
	public int distMode = 0;
	
	public double aErrorSum = 0, bErrorSum = 0;
	public List<double[]> samples;
		
	public void train( int t, double[] x ) {
		DecimalFormat df = new DecimalFormat("00000");
				
		int l = (int)Math.ceil(neurons.size()*ratio);
		sorter.sort(x,neurons);
		
		double[] s_1 = neurons.get(0);
		double[] s_2 = neurons.get(1);
		
		cons.put(new Connection(s_1, s_2),0);
		
		//errors.put(s_1, errors.get(s_1) + distA.dist(s_1, x) );
		if( distMode == 0 ) {
			errors.put(s_1, errors.get(s_1) + Math.pow(distB.dist(s_1, x) ,2) );
		} else {
			errors.put(s_1, errors.get(s_1) + r.nextDouble() );
		}
		
		for( Connection c : cons.keySet() )
			cons.put(c, cons.get(c)+1 );
		
		// train best neuron
		for( int i = 0; i < s_1.length; i++ )
			s_1[i] += lrB * ( x[i] - s_1[i] );
		
		// train neighbors
		for( double[] n : Connection.getNeighbors(cons.keySet(), s_1, 1) )
			for( int i = 0; i < s_1.length; i++ )
				n[i] += lrN * ( x[i] - n[i] );
		
		Set<Connection> consToRemove = new HashSet<Connection>();
		for( Connection c : cons.keySet() )
			if( cons.get(c) > aMax )
				consToRemove.add(c);
		cons.keySet().removeAll(consToRemove);
		
		Set<double[]> neuronsToKeep = new HashSet<double[]>();
		for( Connection c : cons.keySet() ) {
			neuronsToKeep.add(c.getA());
			neuronsToKeep.add(c.getB());
		}
		
		neurons.retainAll(neuronsToKeep);
		errors.keySet().retainAll(neuronsToKeep);
		
		if( t % lambda == 0 ) {
			
			Map<double[],Set<double[]>> map = NGUtils.getBmuMapping(samples, neurons, sorter);
			double aError = DataUtils.getMeanQuantizationError(map, distA);
			double bError = DataUtils.getMeanQuantizationError(map, distB);
						
			double[] q = null;
			double[] f = null;
			
			for( double[] n : neurons )
				if( q == null || errors.get(q) < errors.get(n) ) 
					q = n;
						
			for( double[] n : Connection.getNeighbors(cons.keySet(), q, 1) )
				if( f == null || errors.get(f) < errors.get(n) )
					f = n;
			
			/*for( double[] n : neurons )
				if( q == null || DataUtils.getQuantizationError(q, map.get(q), distB) < DataUtils.getQuantizationError(n, map.get(n), distB ) ) 
					q = n;			
			for( double[] n : Connection.getNeighbors(cons.keySet(), q, 1) )
				if( f == null || DataUtils.getQuantizationError(f, map.get(f), distB) < DataUtils.getQuantizationError(n, map.get(n), distB ) ) 
					f = n;*/
												
			double[] nn = new double[q.length];
			for( int i = 0; i < nn.length; i++ )
				nn[i] = (q[i]+f[i])/2;
			neurons.add(nn);
						
			cons.put( new Connection(q, nn), 0 );
			cons.put( new Connection(f, nn), 0 );
			cons.remove( new Connection( q, f ) );
			
			errors.put(q, errors.get(q) - alpha*errors.get(q) );
			errors.put(f, errors.get(f) - alpha*errors.get(f) );
			//errors.put(nn, (errors.get(q) + errors.get(f))/2);
			errors.put(nn, errors.get(q) );	
			
			// the lower the better
			map = NGUtils.getBmuMapping(samples, neurons, sorter);			
			aErrorSum += DataUtils.getMeanQuantizationError(map, distA) - aError;
			bErrorSum += DataUtils.getMeanQuantizationError(map, distB) - bError;
						
			// draw
			/*Set<double[]> hiliNeurons = new HashSet<double[]>();
			hiliNeurons.add(nn);
			Set<Connection> hiliCons = new HashSet<Connection>();
			hiliCons.add(new Connection(q, nn));
			hiliCons.add(new Connection(f, nn));
			geoDrawNG2("output/"+df.format(t)+".png", neurons, hiliNeurons, cons, hiliCons, new int[]{0,1} );*/
		}
			
		for( double[] n : neurons )
			errors.put(n, errors.get(n) - beta*errors.get(n) );
	}
		
	public List<double[]> getNeurons() {
		return neurons;
	}
	
	public Map<Connection, Integer> getConections() {
		return cons;
	}
	
	private static void geoDrawNG2(String fn, List<double[]> neurons, Set<double[]> hiliNeurons, Map<Connection, Integer> conections, Set<Connection> hiliCons, int[] ga ) {
		int xScale = 1000;
		int yScale = 800;
		
		BufferedImage bufImg = new BufferedImage(xScale, yScale, BufferedImage.TYPE_INT_ARGB);
		Graphics2D g2 = bufImg.createGraphics();
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

		// for scaling
		double maxX = Double.MIN_VALUE;
		double minX = Double.MAX_VALUE;
		double maxY = Double.MIN_VALUE;
		double minY = Double.MAX_VALUE;

		for (double[] n : neurons ) {
			double x = n[ga[0]];
			double y = n[ga[1]];

			if (x > maxX)
				maxX = x;
			if (y > maxY)
				maxY = y;
			if (x < minX)
				minX = x;
			if (y < minY)
				minY = y;
		}

		for( Connection c : conections.keySet() ) {
			if( hiliCons != null && hiliCons.contains(c) )
				g2.setColor(Color.RED);
			else
				g2.setColor(Color.BLACK);
			double[] a = c.getA();
			double[] b = c.getB();
			int x1 = (int)(xScale * (a[ga[0]] - minX)/(maxX-minX));
			int y1 = (int)(yScale * (a[ga[1]] - minY)/(maxY-minY));
			int x2 = (int)(xScale * (b[ga[0]] - minX)/(maxX-minX));
			int y2 = (int)(yScale * (b[ga[1]] - minY)/(maxY-minY));
			g2.drawLine(x1,y1,x2,y2);
		}
		
		for( double[] n : neurons ) {
			if( hiliNeurons.contains(n))
				g2.setColor(Color.RED);
			else
				g2.setColor(Color.BLACK);
			int x1 = (int)(xScale * (n[ga[0]] - minX)/(maxX-minX));
			int y1 = (int)(yScale * (n[ga[1]] - minY)/(maxY-minY));
			g2.fillOval( x1 - 5, y1 - 5, 10, 10	);
		}
		g2.dispose();

		try {
			ImageIO.write(bufImg, "png", new FileOutputStream(fn));
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}
}
