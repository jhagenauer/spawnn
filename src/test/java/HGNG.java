

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
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

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.Connection;
import spawnn.utils.DataUtils;

public class HGNG { // hierarchical growing neural gas
		
	private static Logger log = Logger.getLogger(HGNG.class);
	
	protected List<double[]> neurons = null;
	protected double lrB, lrN;
	protected Map<Connection,Integer> cons;
	protected Dist<double[]> dist;
	protected int aMax;
		
	public HGNG( Collection<double[]> neurons, double lrB, double lrN, Dist<double[]> dist, int aMax ) {
		this.lrB = lrB;
		this.lrN = lrN;
		this.dist = dist;
		this.cons = new HashMap<Connection,Integer>();
		this.aMax = aMax;
				
		this.neurons = new ArrayList<double[]>(neurons);
	}
		
	public void train( int t, double[] x ) {
		
		double[] s_1 = null;
		for( double[] n : neurons ) 
			if( s_1 == null || dist.dist(n, x) < dist.dist(s_1, x) )
				s_1 = n;
						
		double[] s_2 = null;
		for( double[] n : neurons )
			if( n != s_1 && ( s_2 == null || dist.dist(n, x) < dist.dist(s_2, x) ) )
				s_2 = n;

		if( s_2 != null )
			cons.put( new Connection(s_1, s_2),0 );
						
		for( Connection c : cons.keySet() )
			cons.put(c, cons.get(c)+1 );
	
		for( int i = 0; i < s_1.length; i++ )
			s_1[i] += lrB * ( x[i] - s_1[i] );
		
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
		if( !neuronsToKeep.isEmpty() ) // keep at least one
			neurons.retainAll(neuronsToKeep);
		
	}
		
	public List<double[]> getNeurons() {
		return neurons;
	}
	
	public Map<Connection, Integer> getConections() {
		return cons;
	}
	
	public Map<double[],Set<double[]>> getMapping(List<double[]> samples ) {
		Map<double[],Set<double[]>> mapping = new HashMap<double[],Set<double[]>>();
		for( double[] d : samples ) {
			double[] best = null;
			for( double[] n : neurons )
				if( best == null || dist.dist( n, d) < dist.dist( best, d) )
					best = n;
			if( !mapping.containsKey(best))
				mapping.put( best, new HashSet<double[]>() );
			mapping.get(best).add(d);
		}
		return mapping;
	}
	
	static Random r = new Random();
	static double lambda = 1000;
	static double tau_1 = 0.5, tau_2 = 0.25;
		
	public static void main(String[] args) {
										
		List<double[]> samples = DataUtils.readCSV( "data/iris.csv");					
		Dist<double[]> fDist = new EuclideanDist();

		List<double[]> initNeurons = new ArrayList<double[]>();
		for( int i = 0; i < 1; i++ )
			initNeurons.add( Arrays.copyOf(samples.get(i),samples.get(i).length ) );
		
		HGNG ng = new HGNG( initNeurons, 0.05, 0.0005, fDist, 100 );	
		for( int t = 0; t < lambda; t++ ) {
			double[] x = samples.get(r.nextInt(samples.size()));
			ng.train( t, x );
		}
			
		double mqe_0 = DataUtils.getQuantizationError( ng.getNeurons().get(0), samples, fDist );
		log.debug("mqe_0: "+mqe_0);
		
		Map<double[],List<double[]>> tree = new HashMap<double[],List<double[]>>(); // parent, cur neurons
		tree.put( null, ng.getNeurons() );
		
		trainHierarchical( samples, fDist, mqe_0, mqe_0, tree, ng.getNeurons().get(0) );
		
		log.debug("tree:size: "+tree.size());
		for( double[] n : tree.keySet() )
			if( n != null )
				log.debug(Arrays.toString(n)+"->"+tree.get(n));
			else
				log.debug("null"+"->"+tree.get(n));
		
		StringBuffer sb = new StringBuffer();
		expandTreePstree(sb, tree, ng.getNeurons().get(0) );
				
		BufferedWriter w = null;
		
		try {
			w = new BufferedWriter( new FileWriter("output/tree.tex"));
			w.write(sb.toString());
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
		
	private static void expandTreeNewick(StringBuffer sb, Map<double[], List<double[]>> tree, double[] cur ) {
		if( tree.containsKey(cur) ) {
			sb.append("(");
			
			List<double[]> l = tree.get(cur);
			for( int i = 0; i < l.size(); i++ ) {	
				expandTreeNewick(sb, tree, l.get(i) );
				if( i < l.size()-1 )
					sb.append(",");
			}
			sb.append(")");
		}
				
		sb.append(cur.hashCode());
	}
	
	private static void expandTreePstree(StringBuffer sb, Map<double[], List<double[]>> tree, double[] cur ) {
		sb.append("\\pstree{\\TC}{");
		if( tree.containsKey(cur) ) {
			List<double[]> l = tree.get(cur);
			for( int i = 0; i < l.size(); i++ ) {	
				expandTreePstree(sb, tree, l.get(i) );
				
			}
		}
		sb.append("}");	
	}

	public static void trainHierarchical( List<double[]> samples, Dist<double[]> fDist, final double mqe_u, final double mqe_0, Map<double[], List<double[]>> tree, double[] parent ) {
		log.debug("training hierachical");
		
		List<double[]> initNeurons = new ArrayList<double[]>();
		for( int i = 0; i < 2; i++ )
			initNeurons.add( Arrays.copyOf(samples.get(i),samples.get(i).length ) );
		
		HGNG ng = new HGNG( initNeurons, 0.05, 0.0005, fDist, 200 );
		while( true ) {
			
			for( int t = 0; t < lambda; t++ ) {
				double[] x = samples.get(r.nextInt(samples.size()));
				ng.train( t, x );
			}
					
			Map<double[],Set<double[]>> mapping = ng.getMapping(samples);
							
			Map<double[],Double> qes = new HashMap<double[],Double>();
			for( double[] n : ng.getNeurons() )
				if( mapping.containsKey(n) ) 
					qes.put( n, DataUtils.getQuantizationError( n, mapping.get(n), fDist) );
				else
					qes.put( n, 0.0 );
							
			double MQE_m = 0;
			for( double d : qes.values() )
				MQE_m += d;
			MQE_m /= qes.size();
						
			log.debug("1ct: "+MQE_m+" < "+ (tau_1 * mqe_u)+","+ng.getNeurons().size() );
			if( MQE_m < tau_1 * mqe_u ) 
				break;
			
			// grow horizontally
			Map<Connection,Integer> cons = ng.getConections();
			List<double[]> neurons = ng.getNeurons();
			
			double[] q = null;
			for( double[] n : ng.getNeurons() )
				if( q == null || qes.get(q) < qes.get(n) ) 
					q = n;
			
			double[] f = null;
			for( double[] n : Connection.getNeighbors( cons.keySet(), q, 1) )
				if( f == null || qes.get(f) < qes.get(n) )
					f = n;
						
			double[] nn = new double[q.length];
			for( int i = 0; i < nn.length; i++ )
				nn[i] = (q[i]+f[i])/2;
			neurons.add(nn);
						
			cons.put( new Connection(q, nn), 0 );
			cons.put( new Connection(f, nn), 0 );
			cons.remove( new Connection( q, f ) );	
		}
		
		log.debug("training finished. neurons: "+ng.getNeurons().size() );
		tree.put( parent, ng.getNeurons() );
		
		// check all prototypes and train hierarchical
		Map<double[],Set<double[]>> mapping = ng.getMapping(samples);
		for( double[] n : mapping.keySet() ) {
			double mqe_i = DataUtils.getQuantizationError( n, mapping.get(n), fDist);
			log.debug("2ct: "+mqe_i+" < "+ (tau_2 * mqe_0) );
			if( !( mqe_i < tau_2 * mqe_0 ) )
				trainHierarchical( new ArrayList<double[]>(mapping.get(n)), fDist, mqe_i, mqe_0, tree, n);
		}
	}
}
