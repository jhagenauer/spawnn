package spawnn.rbf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import spawnn.SupervisedNet;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.Clustering;
import spawnn.utils.DataUtils;

public class RBF implements SupervisedNet {

	private static Logger log = Logger.getLogger(RBF.class);

	protected List<Map<double[], Double>> weights;
	protected Map<double[], Double> hidden;
	protected Dist<double[]> dist;
	protected double delta;

	public RBF(Map<double[], Double> hidden, int out, Dist<double[]> dist, double delta) {
		this.hidden = hidden;
		this.dist = dist;
		this.delta = delta;

		// init weights
		this.weights = new ArrayList<Map<double[], Double>>();
		for (int i = 0; i < out; i++) {
			Map<double[], Double> m = new HashMap<double[], Double>();
			for (double[] c : hidden.keySet())
				m.put(c, 1.0);
			this.weights.add(m);
		}
	}

	public double[] present(double[] x) {
		double[] response = new double[weights.size()];
		for (double[] c : hidden.keySet()) {
			double output = Math.exp( -0.5 * Math.pow(dist.dist(x, c) / hidden.get(c), 2) );
			for (int i = 0; i < weights.size(); i++)
				response[i] += weights.get(i).get(c) * output;
		}
		return response;
	}
	
	public double[] getResponse( double[] x, double[] neuron ) {
		double output = Math.exp( -0.5 * Math.pow(dist.dist(x, neuron) / hidden.get(neuron), 2) );
		
		double[] response = new double[weights.size()];
		for (int i = 0; i < weights.size(); i++)
			response[i] += weights.get(i).get(neuron) * output;
		return response;
	}
	
	public void train( double t, double[] x, double[] desired ) {
		 train(x, desired);
	}

	public void train(double[] x, double[] desired) {
		
		if( desired.length != weights.size() )
			throw new RuntimeException();

		double[] response = present(x);
		for (double[] c : hidden.keySet()) { // j			
			double z = Math.exp(-0.5 * Math.pow(dist.dist(x, c) / hidden.get(c), 2)); // z_j
			for (int i = 0; i < weights.size(); i++) // i
				weights.get(i).put(c, weights.get(i).get(c) + delta * (desired[i] - response[i]) * z); // delta-rule
		}
	}
	
	public Map<double[],Double> getNeurons() {
		return hidden;
	}
	
	public List<Map<double[], Double>> getWeights() {
		return weights;
	}

	public static void main(String[] args) {
		Random r = new Random();

		List<double[]> samples = DataUtils.readCSV("data/polynomial.csv", new int[] { 5 });
		List<double[]> desired = DataUtils.readCSV("data/polynomial.csv", new int[] { 0, 1, 2, 3, 4 });
		Dist<double[]> dist = new EuclideanDist();
		
		Map<double[], Double> hidden = new HashMap<double[], Double>();

		Map<double[], Set<double[]>> clustering = Clustering.kMeans(samples, 10, dist);
		double qe = DataUtils.getMeanQuantizationError(clustering, dist);
		for( int i = 0; i < 100; i++ ) {
			Map<double[], Set<double[]>> tmp = Clustering.kMeans(samples, clustering.size(), dist);
			if( DataUtils.getMeanQuantizationError(tmp, dist) < qe ) {
				qe = DataUtils.getMeanQuantizationError(tmp, dist);
				clustering = tmp;
			}
		}
		
		for (double[] c : clustering.keySet()) {
			double d = Double.MAX_VALUE;
			for (double[] n : clustering.keySet())
				if (c != n)
					d = Math.min(d, dist.dist(c, n));
			hidden.put(c, d);
		}
		
		RBF rbf = new RBF(hidden, 1, dist, 0.05);

		for (int i = 0; i < 50000; i++) {
			int j = r.nextInt(samples.size());
			rbf.train(samples.get(j), desired.get(j));
		}
		
		List<double[]> response = new ArrayList<double[]>();
		for (double[] x : samples)
			response.add(rbf.present(x));
		
		//log.debug("rmse: "+Meuse.getRMSE(response, desired) ); 
		//log.debug("r^2: "+Math.pow(Meuse.getPearson(response, desired), 2) ); 
	}
	
	public String toString() {
		StringBuffer sb = new StringBuffer();
		for( double[] d : hidden.keySet() ) {
			sb.append(Arrays.toString(d)+" ("+hidden.get(d)+"): ");
			for( int i = 0; i < weights.size(); i++ ) {
				sb.append(i+","+weights.get(i).get(d)+",");
			}
			sb.append("\n");
		}	
		return sb.toString();
	}
}
