package growing_cng;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import javax.imageio.ImageIO;

import spawnn.dist.Dist;
import spawnn.ng.Connection;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.utils.DataUtils;

public class GrowingCNG {
	
	protected List<double[]> neurons = null;
	protected double lrB, lrN, alpha, beta, ratio;
	protected Map<Connection,Integer> cons;
	protected Dist<double[]> distA, distB;
	protected Map<double[],Double> errors;
	protected int aMax, lambda;
	protected Random r = new Random();
	protected int maxNeurons;
	
	public GrowingCNG( List<double[]> neurons, double lrB, double lrN, Dist<double[]> distA, Dist<double[]> distB, double ratio, int aMax, int lambda, double alpha, double beta ) {
		this(neurons,lrB,lrN,distA,distB,ratio,aMax,lambda,alpha,beta,Integer.MAX_VALUE);
	}
				
	public GrowingCNG( List<double[]> neurons, double lrB, double lrN, Dist<double[]> distA, Dist<double[]> distB, double ratio, int aMax, int lambda, double alpha, double beta, int maxNeurons ) {
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
		this.maxNeurons = maxNeurons;
		this.neurons = new ArrayList<double[]>(neurons);
		
		this.errors = new HashMap<double[],Double>();
		for( double[] n : this.neurons ) {
			this.errors.put( n, 0.0 );
		}
	}
	
	public int distMode = 0;
	public double aErrorSum = 0, bErrorSum = 0;
	public int k = 0;
	public List<double[]> samples;
	public int run = 0;
		
	public void train( int t, double[] x ) {
		int[] stats = sortNeurons(x);
		
		// find idxF1
		double[] f1 = null;
		int idxF1 = -1;
		for( int i = 0; i < neurons.size(); i++ ) {
			double[] n = neurons.get(i);
			if( f1 == null || distB.dist(n,x) < distB.dist(f1, x) ) {
				f1 = n;
				idxF1 = i;
			}
		}
		int idxG1 = stats[0];
		int idxF2 = stats[1];
		int l = (int)stats[2];
						
		double[] s_1 = neurons.get(0);
		double[] s_2 = neurons.get(1);
		
		cons.put(new Connection(s_1, s_2),0);
		if( distMode == 0 ) 
			errors.put( s_1, errors.get(s_1) + r.nextDouble() );
		else if( distMode == 1)
			errors.put(s_1, errors.get(s_1) + distA.dist(s_1, x) );
		else if( distMode == 2 )
			errors.put(s_1, errors.get(s_1) + distB.dist(s_1, x) );		
		else if( distMode == 4 ) 
			errors.put( s_1, errors.get(s_1) + ratio*idxF1/neurons.size() );
		else if( distMode == 5 ) 
			errors.put( s_1, errors.get(s_1) + ratio*(idxF1+1)/neurons.size() ); // deutlich besser als 4... Warum?!
		else if( distMode == 6 ) 
			errors.put( s_1, errors.get(s_1) + ratio*(idxF2+1)/l );
		else if( distMode == 7 ) 
			errors.put( s_1, errors.get(s_1) + (1.0-ratio)*(idxG1+1)/l + ratio*(idxF2+1)/l );
		else if( distMode == 8 ) 
			errors.put( s_1, errors.get(s_1) + (1.0-ratio)*(idxG1+1)/l + ratio*(idxF1+1)/neurons.size() );
		
		
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
		
		if( t % lambda == 0 ) 
			insertNeuron(t);
					
		for( double[] n : neurons )
			errors.put(n, errors.get(n) - beta*errors.get(n) );
	}
	
	public void insertNeuron(int t) {
		Map<double[],Set<double[]>> map = getMapping(samples);
		double preA = DataUtils.getMeanQuantizationError(map, distA);
		double preB = DataUtils.getMeanQuantizationError(map, distB);
					
		double[] q = null;
		double[] f = null;
		
		for( double[] n : neurons )
			if( q == null || errors.get(q) < errors.get(n) ) 
				q = n;
					
		for( double[] n : Connection.getNeighbors(cons.keySet(), q, 1) )
			if( f == null || errors.get(f) < errors.get(n) )
				f = n;
		
		if( distMode == -1 ) {
			q = neurons.get(r.nextInt(neurons.size()));
			List<double[]> nbs = new ArrayList<double[]>( Connection.getNeighbors(cons.keySet(), q, 1));
			f = nbs.get(r.nextInt(nbs.size()));
		}
							
		double[] nn = new double[q.length];
		for( int i = 0; i < nn.length; i++ )
			nn[i] = (q[i]+f[i])/2;
		neurons.add(nn);
								
		cons.put( new Connection(q, nn), 0 );
		cons.put( new Connection(f, nn), 0 );
		cons.remove( new Connection( q, f ) );
		
		errors.put(q, errors.get(q) - alpha * errors.get(q) );
		errors.put(f, errors.get(f) - alpha * errors.get(f) );
		errors.put(nn, (errors.get(q) + errors.get(f))/2);
		//errors.put(nn, errors.get(q) );
		
		if( t > 60000 ) {
			map = getMapping(samples);
			aErrorSum += DataUtils.getMeanQuantizationError(map, distA) - preA;
			bErrorSum += DataUtils.getMeanQuantizationError(map, distB) - preB;
			k++;
		}		
		
		// draw
		/*DecimalFormat df = new DecimalFormat("000000");
		Set<double[]> hiliNeurons = new HashSet<double[]>();
		hiliNeurons.add(nn);
		Set<Connection> hiliCons = new HashSet<Connection>();
		hiliCons.add(new Connection(q, nn));
		hiliCons.add(new Connection(f, nn));
		geoDrawNG2("output/"+distMode+"_"+df.format(t)+"_"+run+".png", neurons, hiliNeurons, cons, hiliCons, new int[]{0,1}, samples );*/
	}
			
	public List<double[]> getNeurons() {
		return neurons;
	}
	
	public Map<Connection, Integer> getConections() {
		return cons;
	}
	
	private int[] sortNeurons(double[] x) {
		int l = (int)Math.round( ratio * ( neurons.size() -1 ) + 1 );
		new DefaultSorter<double[]>(distA).sort(x, neurons);
		List<double[]> sortedA = new ArrayList<double[]>(neurons);
				
		new DefaultSorter<double[]>(distB).sort(x, neurons.subList(0, l));
		
		int idxG1 = neurons.indexOf(sortedA.get(0));
		int idxF2 = sortedA.indexOf(neurons.get(0));
		return new int[]{idxG1, idxF2, l};
	}
	
	public Map<double[],Set<double[]>> getMapping( List<double[]> samples ) {
		Map<double[],Set<double[]>> r = new HashMap<double[],Set<double[]>>();
		for( double[] x : samples ) {
			sortNeurons(x);
			double[] bmu = neurons.get(0);
			if( !r.containsKey(bmu) )
				r.put(bmu, new HashSet<double[]>() );
			r.get(bmu).add(x);
		}
		return r;
	}
	
	public static void geoDrawNG2(String fn, List<double[]> neurons, Set<double[]> hiliNeurons, Map<Connection, Integer> conections, Set<Connection> hiliCons, int[] ga, List<double[]> samples ) {
		int xScale = 1000;
		int yScale = 800;
		
		BufferedImage bufImg = new BufferedImage(xScale, yScale, BufferedImage.TYPE_INT_RGB);
		Graphics2D g2 = bufImg.createGraphics();
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		g2.fillRect(0, 0, xScale, yScale);

		// for scaling
		double maxX = Double.MIN_VALUE;
		double minX = Double.MAX_VALUE;
		double maxY = Double.MIN_VALUE;
		double minY = Double.MAX_VALUE;

		for (double[] n : samples ) {
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
		
		for( double[] n : samples ) {
			g2.setColor(Color.GRAY);
			int x1 = (int)(xScale * (n[ga[0]] - minX)/(maxX-minX));
			int y1 = (int)(yScale * (n[ga[1]] - minY)/(maxY-minY));
			g2.fillOval( x1 - 3, y1 - 3, 6, 6	);
		}

		if( conections != null )
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
			if( hiliNeurons != null && hiliNeurons.contains(n))
				g2.setColor(Color.RED);
			else
				g2.setColor(Color.BLACK);
			int x1 = (int)(xScale * (n[ga[0]] - minX)/(maxX-minX));
			int y1 = (int)(yScale * (n[ga[1]] - minY)/(maxY-minY));
			g2.fillOval( x1 - 5, y1 - 5, 10, 10	);
			g2.drawString(""+n.hashCode(),x1,y1);
		}
		g2.dispose();

		try {
			ImageIO.write(bufImg, "png", new FileOutputStream(fn));
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}
}
