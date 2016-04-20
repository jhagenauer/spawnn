

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

import com.vividsolutions.jts.geom.Geometry;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.Connection;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.RegionUtils;
import spawnn.utils.SpatialDataFrame;

/* Just an experiment, connections are created based on contiguity
 * Presumable result: Resulting topology reflects conectivity of cluster.
 */

public class GeoGNG { 

	private static Logger log = Logger.getLogger(GeoGNG.class);

	protected List<double[]> neurons = null;
	protected double lrB, lrN, alpha, beta;
	protected Map<Connection, Integer> cons;
	protected Map<double[],Double> errors;
	protected Dist<double[]> dist;
	protected int aMax;
	
	protected Map<double[],double[]> hist; // sample to neuron
	protected Map<double[], Set<double[]>> ctg;

	public GeoGNG(Collection<double[]> neurons, double lrB, double lrN, Dist<double[]> dist, int aMax, Map<double[], Set<double[]>> ctg, double alpha, double beta ) {
		this.lrB = lrB;
		this.lrN = lrN;
		this.dist = dist;
		this.cons = new HashMap<Connection, Integer>();
		this.hist = new HashMap<double[],double[]>();
		this.aMax = aMax;
		this.ctg = ctg;
		this.alpha = alpha;
		this.beta = beta;

		this.neurons = new ArrayList<double[]>(neurons);
		
		this.errors = new HashMap<double[],Double>();
		for( double[] n : this.neurons )
			this.errors.put( n, 0.0 );
	}

	public void train(int t, double[] x) {

		// get best s_1
		double[] s_1 = null;
		for (double[] n : neurons)
			if (s_1 == null || dist.dist(n, x) < dist.dist(s_1, x))
				s_1 = n;
		
		hist.put( x, s_1 );
		
		// get all neurons that map neighbor of x and are not s_1
		Set<double[]> nbs = new HashSet<double[]>();
		for( double[] d : ctg.get(x) ) {
			if( hist.containsKey(d) && hist.get(d) != s_1 && neurons.contains( hist.get(d) ) ) 
				nbs.add( hist.get(d) );
		}
				
		for( double[] d : nbs )
			cons.put(new Connection(s_1, d), 0);
		
		// update error
		errors.put(s_1, errors.get(s_1)+dist.dist(s_1, x) );
		
		// move to x
		for (int i = 0; i < s_1.length; i++)
			s_1[i] += lrB * (x[i] - s_1[i]);

		for (double[] n : Connection.getNeighbors(cons.keySet(), s_1, 1 ) )
			for (int i = 0; i < s_1.length; i++)
				n[i] += lrN * (x[i] - n[i]);
		
		// increase age		
		for (Connection c : cons.keySet())
			cons.put(c, cons.get(c) + 1);

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
		
		if( neuronsToRetain.isEmpty() ) {
			neuronsToRetain.add( neurons.get(0) );
			neuronsToRetain.add( neurons.get(1) );
			cons.put( new Connection(neurons.get(0) , neurons.get(1) ), 0);
		}
	
		neurons.retainAll(neuronsToRetain);
		errors.keySet().retainAll(neuronsToRetain);
		
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

	public Map<double[], Set<double[]>> getMapping(List<double[]> samples) {
		Map<double[], Set<double[]>> mapping = new HashMap<double[], Set<double[]>>();
		for (double[] d : samples) {
			double[] best = null;
			for (double[] n : neurons)
				if (best == null || dist.dist(n, d) < dist.dist(best, d))
					best = n;
			if (!mapping.containsKey(best))
				mapping.put(best, new HashSet<double[]>());
			mapping.get(best).add(d);
		}
		return mapping;
	}

	static Random r = new Random();
	static double lambda = 1000;
	static double tau_1 = 0.5, tau_2 = 0.25;

	public static void main(String[] args) {
				
		File file = new File("data/redcap/Election/election2004.shp");
		SpatialDataFrame sd = DataUtils.readShapedata(file, new int[]{}, false );
		List<double[]> samples = sd.samples;	
		List<Geometry> geoms = sd.geoms;
		
		Map<double[], Set<double[]>> ctg = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");
				
		Dist<double[]> fDist = new EuclideanDist( new int[]{7});

		List<double[]> initNeurons = new ArrayList<double[]>();
		for (int i = 0; i < 2; i++)
			initNeurons.add(Arrays.copyOf(samples.get(i), samples.get(i).length));
		
		GeoGNG ng = new GeoGNG(initNeurons, 0.05, 0.0005, fDist, 200, ctg, 0.5, 0.0005 );

		for( int i = 0; i < 10000; i++ ) {
			double[] x = samples.get(r.nextInt(samples.size()));
			ng.train(i, x);
		}
		
		log.debug("Neurons: "+ng.getNeurons().size() );
		Map<double[], Set<double[]>> mapping = ng.getMapping(samples);
		Drawer.geoDrawCluster(mapping.values(), samples, geoms, "output/cluster.png", true);
	}
}
