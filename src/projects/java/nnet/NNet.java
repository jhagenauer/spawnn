package nnet;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import nnet.activation.Constant;
import nnet.activation.Function;
import nnet.activation.Identity;
import nnet.activation.Sigmoid;
import rbf.Meuse;
import spawnn.SupervisedNet;
import spawnn.dist.EuclideanDist;
import spawnn.rbf.RBF;
import spawnn.utils.Clustering;
import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.transform;

public class NNet implements SupervisedNet {

	private static Logger log = Logger.getLogger(NNet.class);

	private Function[][] layer;
	private double[][][] weights;
	private double eta = 0.01;

	public NNet(Function[][] layer, double[][][] weights) {
		this.layer = layer;
		this.weights = weights;
	}

	@Override
	public double[] present(double[] x) {
		double[] in = x;
		for (int i = 0;; i++) {

			double[] o = new double[layer[i].length];
			// apply activation
			for (int j = 0; j < layer[i].length; j++)
				if (j < in.length) // extra-neurons are bias
					o[j] = layer[i][j].f(in[j]);
				else {
					o[j] = layer[i][j].f(1.0);
				}

			if (i == layer.length - 1)
				return o;

			// apply weights, calculate new input
			in = new double[weights[i][0].length]; // assumes first neuron is connected to all non-bias neurons
			for (int j = 0; j < o.length; j++)
				for (int k = 0; k < in.length; k++)
					in[k] += weights[i][j][k] * o[j];
		}
	}

	@Override
	public void train(double t, double[] x, double[] desired) {
		double[][] out = new double[layer.length][];

		// forward propagation
		double[] in = x;
		for (int i = 0;; i++) {

			out[i] = new double[layer[i].length];
			// apply activation
			for (int j = 0; j < layer[i].length; j++)
				if (j < in.length) // extra-neurons are bias
					out[i][j] = layer[i][j].f(in[j]);
				else {
					out[i][j] = layer[i][j].f(1.0);
				}

			if (i == layer.length - 1)
				break;
			
			// apply weights, calculate new net input
			in = new double[weights[i][0].length]; // assumes first neuron is connected to all non-bias neurons
			for (int j = 0; j < out[i].length; j++)
				for (int k = 0; k < in.length; k++)
					in[k] += 
					weights[i][j][k] *	
					out[i][j];
		}

		// back propagation
		double[][] error = new double[layer.length][];
		int ll = layer.length - 1; // last layer

		// get error
		error[ll] = new double[layer[ll].length];
		for (int j = 0; j < layer[ll].length; j++)
			error[ll][j] = (out[ll][j] - desired[j]) * out[ll][j] * (1.0 - out[ll][j]);

		// change weights
		for (int j = 0; j < weights[ll - 1].length; j++)
			for (int k = 0; k < weights[ll - 1][j].length; k++)
				weights[ll - 1][j][k] -= eta * error[ll][k] * out[ll - 1][j];

		// i gues there is a problem here
		/*for (int i = ll - 1; i > 0; i--) {
			// get error
			error[i] = new double[layer[i].length];
			for (int j = 0; j < layer[i].length; j++) {
				double s = 0;
				for (int k = 0; k < weights[i][j].length; k++)
					s += error[i + 1][k] * weights[i][j][k];
				error[i][j] = s * out[i][j] * (1.0 - out[i][j]);
			}

			// change weights
			for (int j = 0; j < weights[i - 1].length; j++)
				for (int k = 0; k < weights[i - 1][j].length; k++)
					weights[i - 1][j][k] -= eta * error[i][k] * out[i - 1][j];
		}*/
	}
	
	private void printWeights(int layer) {
		System.out.println("layer: "+layer);
		for( int i = 0; i < weights[layer].length; i++ )
			for( int j = 0; j < weights[layer][i].length; j++ )
				System.out.println(i+" --> "+j+", "+weights[layer][i][j]);
		
	}
	

	/*
	 * not really useful for feed forward network, also weight is not compatible with weights, needs switching of axis
	 */
	@Override
	public double[] getResponse(double[] x, double[] weight) {
		double[] r = new double[x.length];
		for (int i = 0; i < x.length; i++)
			r[i] = x[i] * weight[i];
		return r;
	}

	public static void main(String[] args) {
		Random r = new Random(0);

		DataFrame df = DataUtils.readDataFrameFromCSV(new File("data/polynomial.csv"), new int[] {}, true);
		for (int i = 0; i < df.names.size(); i++)
			log.debug(i + ":" + df.names.get(i));
		DataUtils.transform(df.samples, transform.zScore);

		List<double[]> samplesTrain = new ArrayList<>();
		List<double[]> desiredTrain = new ArrayList<>();
		List<double[]> samplesVal = new ArrayList<>();
		List<double[]> desiredVal = new ArrayList<>();

		for (double[] d : df.samples) {
			if (samplesTrain.size() < df.samples.size() * 2.0 / 3.0) {
				samplesTrain.add(Arrays.copyOf(d, 4));
				desiredTrain.add(new double[] { d[5] });
			} else {
				samplesVal.add(Arrays.copyOf(d, 4));
				desiredVal.add(new double[] { d[5] });
			}
		}

		{ // RBFN-Test
			EuclideanDist fDist = new EuclideanDist(new int[] { 0, 1, 2, 3 });
			Map<double[], Set<double[]>> map = Clustering.kMeans(df.samples, 10, fDist);

			Map<double[], Double> hidden = new HashMap<double[], Double>();
			// min plus overlap
			for (double[] c : map.keySet()) {
				double d = Double.MAX_VALUE;
				for (double[] n : map.keySet())
					if (c != n)
						d = Math.min(d, fDist.dist(c, n)) * 1.1;
				hidden.put(c, d);
			}
			RBF rbf = new RBF(hidden, 1, fDist, 0.1);
			for (int i = 0; i < 100000; i++) {
				int idx = r.nextInt(samplesTrain.size());
				rbf.train(samplesTrain.get(idx), desiredTrain.get(idx));
			}

			List<double[]> response = new ArrayList<double[]>();
			for (double[] x : samplesVal)
				response.add(rbf.present(x));
			log.debug("RMSE RBF: " + Meuse.getRMSE(response, desiredVal));

		}

		Function[] input = new Function[] { 
				new Identity(), 
				new Identity(), 
				new Identity(), 
				new Identity(), 
				new Constant(1.0) // Bias
		};

		Function[] hidden = new Function[] { 
				new Sigmoid(), 
				new Sigmoid(), 
				new Constant(1.0) // Bias
		};

		Function[] output = new Function[] { new Identity() };

		Function[][] layer = new Function[][] { input, hidden, output };

		// connections, no connection to last neuron of hidden layer (bias neuron)
		double[][] w1 = new double[input.length][hidden.length - 1];
		for (int i = 0; i < input.length; i++)
			for (int j = 0; j < hidden.length - 1; j++)
				w1[i][j] = r.nextDouble();

		double[][] w2 = new double[hidden.length][output.length];
		for (int i = 0; i < w2.length; i++)
			for (int j = 0; j < w2[i].length; j++)
				w2[i][j] = r.nextDouble();

		double[][][] weights = new double[][][] { w1, w2 };

		NNet nnet = new NNet(layer, weights);

		List<double[]> response = new ArrayList<double[]>();
		for (double[] x : samplesVal)
			response.add(nnet.present(x));
		log.debug("RMSE pre-train: " + Meuse.getRMSE(response, desiredVal));

		for (int i = 0; i < 100000; i++) {
			int idx = r.nextInt(samplesTrain.size());
			nnet.train(-1, samplesTrain.get(idx), desiredTrain.get(idx));
		}

		response = new ArrayList<double[]>();
		for (double[] x : samplesVal)
			response.add(nnet.present(x));
		log.debug("RMSE post-train: " + Meuse.getRMSE(response, desiredVal));
	}
}
