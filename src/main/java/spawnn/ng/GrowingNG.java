package spawnn.ng;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
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

import spawnn.UnsupervisedNet;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.utils.DataUtils;

public class GrowingNG implements UnsupervisedNet {
	
	private static Logger log = Logger.getLogger(GrowingNG.class);
	
	protected List<double[]> neurons = null;
	protected double lrB, lrN, alpha, beta;
	protected Sorter<double[]> sorter;
	protected Map<Connection,Integer> cons;
	protected Dist<double[]> dist;
	protected Map<double[],Double> errors;
	protected int aMax, lambda;
		
	public GrowingNG( Collection<double[]> neurons, double lrB, double lrN, Sorter<double[]> sorter, Dist<double[]> dist, int aMax, int lambda, double alpha, double beta ) {
		this.lrB = lrB;
		this.lrN = lrN;
		this.dist = dist;
		this.cons = new HashMap<Connection,Integer>();
		this.aMax = aMax;
		this.lambda = lambda;
		this.alpha = alpha;
		this.beta = beta;
		this.sorter = sorter;
		this.neurons = new ArrayList<double[]>(neurons);
		
		this.errors = new HashMap<double[],Double>();
		for( double[] n : this.neurons )
			this.errors.put( n, 0.0 );
	}
		
	public void train( double t, double[] x ) {
		
		sorter.sort(x, neurons);
		double[] s_1 = neurons.get(0);
		double[] s_2 = neurons.get(1);
		
		cons.put(new Connection(s_1, s_2),0);
		
		errors.put(s_1, errors.get(s_1)+dist.dist(s_1, x) );
		
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
			double[] q = null;
			for( double[] n : neurons )
				if( q == null || errors.get(q) < errors.get(n) ) 
					q = n;
			
			double[] f = null;
			for( double[] n : Connection.getNeighbors(cons.keySet(), q, 1) )
				if( f == null || errors.get(f) < errors.get(n) )
					f = n;
						
			double[] nn = new double[q.length];
			for( int i = 0; i < nn.length; i++ )
				nn[i] = (q[i]+f[i])/2;
			neurons.add(nn);
						
			cons.put( new Connection(q, nn), 0 );
			cons.put( new Connection(f, nn), 0 );
			cons.remove( new Connection( q, f ) );
			
			errors.put(q, errors.get(q) - alpha*errors.get(q) );
			errors.put(f, errors.get(f) - alpha*errors.get(f) );
			errors.put(nn, (errors.get(q)+errors.get(f))/2);	
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
	
	public static void main(String[] args) {
		Random r = new Random();
		int T_MAX = 100000;
						
		List<double[]> samples = null;
		try {
			samples = DataUtils.readCSV( new FileInputStream(new File("data/iris.csv") ) );			
			DataUtils.normalize(samples);
			
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
				
		Dist<double[]> eDist = new EuclideanDist();
		
		List<double[]> neurons = new ArrayList<double[]>();
		for( int i = 0; i < 2; i++ ) {
			double[] d = samples.get(i);
			neurons.add( Arrays.copyOf(d, d.length) );
		}
		GrowingNG ng = new GrowingNG(neurons, 0.05, 0.0005, new DefaultSorter<double[]>(eDist), eDist, 90, 300, 0.5, 0.0005);		
		for (int t = 0; t < T_MAX; t++) {
			double[] x = samples.get(r.nextInt(samples.size()));
			ng.train( t, x );
		}
		
		log.debug(ng.getNeurons().size() );
	}
}
