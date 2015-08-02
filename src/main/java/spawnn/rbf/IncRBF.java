package spawnn.rbf;

import java.util.ArrayList;
import java.util.Arrays;
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

public class IncRBF extends RBF {
	
	private static Logger log = Logger.getLogger(IncRBF.class);

	protected double lrA, lrB, alpha, beta;
	protected Map<Connection,Integer> cons;
	protected int aMax;
	public Map<double[],Double> errors; // TODO make it protected
	protected int t = 0;
	private final double minR = Math.pow(10,-127);
	
	public double scale = 1.0; // this is not in the original implementation
		
	public IncRBF( Map<double[], Double> hidden, double lrA, double lrB, Dist<double[]> distA, int aMax, double alpha, double beta, double delta, int out ) {
		super(hidden, out, distA, delta);
		this.lrA = lrA;
		this.lrB = lrB;
		this.cons = new HashMap<Connection,Integer>();
		this.aMax = aMax;
		this.alpha = alpha;
		this.beta = beta;
		
		this.errors = new HashMap<double[],Double>();
		for( double[] n : this.hidden.keySet() )
			this.errors.put( n, 1000.0 );
	}
		
	@Override
	public void train( double[] x, double[] desired ) {
				
		for( double[] n : hidden.keySet() )
			errors.put(n, errors.get(n) - alpha * errors.get(n) );
		
		// get best and second best match		
		double[] s_1 = null;
		for( double[] n : hidden.keySet() ) 
			if( s_1 == null || dist.dist(n, x) < dist.dist(s_1, x) )
				s_1 = n;
				
		double[] s_2 = null;
		for( double[] n : hidden.keySet() )
			if( n != s_1 && ( s_2 == null || dist.dist(n, x) < dist.dist(s_2, x) ) )
				s_2 = n;
		
		// create edge
		cons.put(new Connection(s_1, s_2),0);
		
		// move neurons
		Set<double[]> moved = new HashSet<double[]>();
		for( int i = 0; i < s_1.length; i++ )
			s_1[i] += lrA * ( x[i] - s_1[i] );
		moved.add(s_1);
		
		for( double[] nb : Connection.getNeighbors(cons.keySet(), s_1, 1) ) {
			for( int i = 0; i < s_1.length; i++ )
				nb[i] += lrB * ( x[i] - nb[i] );
			moved.add(nb);
		}
		
		// update radii
		for( double[] n : moved ) {
			Set<double[]> nbs = Connection.getNeighbors(cons.keySet(), n, 1);
			double m = 0;
			for( double[] nb : nbs )
				m += dist.dist(n, nb);
			double nr = Math.max( minR, scale * m/nbs.size() );
			hidden.put( n, nr );
		}
		
		// train weights
		super.train(x, desired);
		
		// update prediction-error
		double[] response = present(x);
		double msq = 0;
		for( int i = 0; i < desired.length; i++ )
			msq += Math.pow( desired[0] - response[0], 2 )/desired.length;
		errors.put( s_1, errors.get(s_1) + msq ); 
		
		// increment age
		for( Connection c : cons.keySet() )
			cons.put(c, cons.get(c)+1 );
				
		// remove to old cons
		Set<Connection> consToRetain = new HashSet<Connection>();
		for (Connection c : cons.keySet())
			if( cons.get(c) <= aMax )
				consToRetain.add(c);		
		cons.keySet().retainAll(consToRetain);
		
		// remove neurons without cons
		Set<double[]> neuronsToRetain = new HashSet<double[]>();
		for( Connection c : cons.keySet() ) {
			neuronsToRetain.add(c.getA());
			neuronsToRetain.add(c.getB());
		}
		
		hidden.keySet().retainAll(neuronsToRetain);
		errors.keySet().retainAll(neuronsToRetain);
		for( Map<double[],Double> m : weights )
			m.keySet().retainAll(neuronsToRetain);
	}
	
	public void insert() {
		double[] q = null;
		for( double[] n : hidden.keySet() )
			if( q == null || errors.get(q) < errors.get(n) ) 
				q = n;
		
		double[] f = null;
		for( double[] n : Connection.getNeighbors(cons.keySet(), q, 1) )
			if( f == null || errors.get(f) < errors.get(n) )
				f = n;
					
		double[] nn = new double[q.length];
		for( int i = 0; i < nn.length; i++ )
			nn[i] = (q[i]+f[i])/2;			
		hidden.put(nn,1.0);
					
		cons.put( new Connection(q, nn), 0 );
		cons.put( new Connection(f, nn), 0 );
		cons.remove( new Connection( q, f ) );
		
		errors.put(q, errors.get(q) - beta * errors.get(q) );
		errors.put(f, errors.get(f) - beta * errors.get(f) );
		errors.put(nn, (errors.get(q) + errors.get(f))/2);
					
		// update weights
		for( Map<double[],Double> m : weights )
			m.put( nn, (m.get(q)+m.get(f))/2 );
	}
		
	public Map<Connection, Integer> getConections() {
		return cons;
	}
	
	public double getTotalError() {
		double sum = 0;
		for( double d : errors.values() )
			sum += d;
		return sum;
	}
	
	public static void main(String[] args) {
		int T_MAX = 100000;
		Random r = new Random();

		List<double[]> samples = DataUtils.readCSV("data/polynomial.csv", new int[] { 5 });
		List<double[]> desired = DataUtils.readCSV("data/polynomial.csv", new int[] { 0, 1, 2, 3, 4 });
		Dist<double[]> dist = new EuclideanDist();
			
		int maxK = 4;
		double mRMSE = 0;
		double mNeurons = 0;
		
		for( int k = 0; k < maxK; k++ ) {
			
			List<double[]> training = new ArrayList<double[]>();
			List<double[]> trainingDesired = new ArrayList<double[]>();
			List<double[]> validation = new ArrayList<double[]>();
			List<double[]> validationDesired = new ArrayList<double[]>();
			
			// k-fold
			for (int i = 0; i < samples.size(); i++) {
				if (k * samples.size() / maxK <= i && i < (k + 1) * samples.size() / maxK) {
					validation.add(samples.get(i));
					validationDesired.add(desired.get(i));
				} else {
					training.add(samples.get(i));
					trainingDesired.add(desired.get(i));
				}
			}
			
			
			Map<double[],Double> hidden = new HashMap<double[],Double>();
			while( hidden.size() < 2 ) {
				double[] d = samples.get(r.nextInt(samples.size()));
				hidden.put( Arrays.copyOf(d, d.length), 1.0 );
			}
			IncRBF irbf = new IncRBF(hidden, 0.05, 0.0005, dist, 50, 0.0005, 0.5, 0.05, 1 );		
			
			for (int t = 0; t < T_MAX; t++) {
				int idx = r.nextInt(samples.size());
				irbf.train( samples.get(idx), desired.get(idx) );
				
				if( t % 5000 == 0 )
					irbf.insert();
			}
			
			double rmse = 0;
			for (int i = 0; i < samples.size(); i++)
				rmse += Math.pow(irbf.present(samples.get(i))[0] - desired.get(i)[0], 2);
			rmse = Math.sqrt(rmse / samples.size());
			
			mRMSE += rmse;
			mNeurons += irbf.getNeurons().size();
		
		}
		
		log.debug("neurons: "+mNeurons/maxK);
		log.debug("rmse: "+mRMSE/maxK);
	}
}
