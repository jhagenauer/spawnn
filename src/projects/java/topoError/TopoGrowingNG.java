package topoError;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import regionalization.RegionUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.Connection;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.Point;

public class TopoGrowingNG {
	
	private static Logger log = Logger.getLogger(TopoGrowingNG.class);
	
	protected List<double[]> neurons = null;
	protected double lrB, lrN, alpha, beta;
	protected Map<Connection,Integer> cons;
	protected Map<Connection,Integer> errors;
	protected Dist<double[]> dist;
	protected int aMax, lambda;
	protected Map<double[], Set<double[]>> ctg;
		
	public TopoGrowingNG( Collection<double[]> neurons, double lrB, double lrN, Dist<double[]> dist, int aMax, int lambda, double alpha, double beta, Map<double[], Set<double[]>> ctg ) {
		this.lrB = lrB;
		this.lrN = lrN;
		this.dist = dist;
		
		this.cons = new HashMap<Connection,Integer>();
		this.errors = new HashMap<Connection,Integer>();
		
		this.aMax = aMax;
		this.lambda = lambda;
		this.alpha = alpha;
		this.beta = beta;
		this.ctg = ctg;
				
		this.neurons = new ArrayList<double[]>(neurons);
	}
		
	public void train( int t, double[] x ) {
		
		// get closest neuron
		double[] s_1 = null;
		for( double[] n : neurons ) 
			if( s_1 == null || dist.dist(n, x) < dist.dist(s_1, x) )
				s_1 = n;
				
		// get topo error
		for( double[] nb : ctg.get(x) ) {
			// get closest to nb
			double[] s_2 = null;
			for( double[] n : neurons ) 
				if( s_2 == null || dist.dist(n, nb) < dist.dist(s_2, nb) )
					s_2 = n;
			
			Connection c = new Connection(s_1, s_2);
			if( !cons.containsKey(c)) // error
				
				if( errors.containsKey(c) )
					errors.put(c, errors.get(c) + 1 );
				else
					errors.put(c, 1);
			else {
				cons.put(c, 0);
				errors.remove(c);
			}
		}				
		
		// age
		for( Connection c : cons.keySet() )
			cons.put(c, cons.get(c)+1 );
		
		// learn
		for( int i = 0; i < s_1.length; i++ )
			s_1[i] += lrB * ( x[i] - s_1[i] );
		
		for( double[] n : Connection.getNeighbors(cons.keySet(), s_1, 1) )
			for( int i = 0; i < s_1.length; i++ )
				n[i] += lrN * ( x[i] - n[i] );
		
		// remove old cons
		Set<Connection> consToRemove = new HashSet<Connection>();
		for( Connection c : cons.keySet() )
			if( cons.get(c) > aMax )
				consToRemove.add(c);
		cons.keySet().removeAll(consToRemove);
		errors.keySet().removeAll(consToRemove);
				
		if( t % lambda == 0 ) { // add connection
			Connection maxErrorC = null;
			for( Connection c : errors.keySet() ) 
				if( maxErrorC == null || errors.get(c) > errors.get(maxErrorC) )
					maxErrorC = c;
			
			cons.put(maxErrorC, 0);
			errors.remove(maxErrorC);
		}	
	}
		
	public List<double[]> getNeurons() {
		return neurons;
	}
	
	public Map<Connection, Integer> getConections() {
		return cons;
	}
	
	public static void main(String[] args) {
		Random r = new Random();
		int T_MAX = 1000000;
						
		File file = new File("data/redcap/Election/election2004.shp");
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(file, true);
		
		List<double[]> samples = sdf.samples;
		List<Geometry> geoms = sdf.geoms;
		Map<double[], Set<double[]>> ctg = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");

		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			Point p1 = geoms.get(i).getCentroid();

			// ugly, but easy
			d[0] = p1.getX();
			d[1] = p1.getY();
		}
		
		final int[] ga = new int[] { 0, 1 };
		final int[] fa = { 7 };
		
		final Dist<double[]> fDist = new EuclideanDist( fa);
		
		List<double[]> neurons = new ArrayList<double[]>();
		for( int i = 0; i < 64; i++ ) {
			double[] d = samples.get(i);
			neurons.add( Arrays.copyOf(d, d.length) );
		}
		
		TopoGrowingNG ng = new TopoGrowingNG(neurons, 0.05, 0.0005, fDist, 400, 4, 0.5, 0.0005, ctg);		
		for (int t = 0; t < T_MAX; t++) {
			double[] x = samples.get(r.nextInt(samples.size()));
			ng.train( t, x );
		}
		
		log.debug(ng.getNeurons().size() );
		log.debug(ng.cons.size());
		Connection.writeGraphML(ng.cons, "output/conns.graphml");
	}
}
