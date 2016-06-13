package nnet;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import nnet.activation.Constant;
import nnet.activation.Function;
import nnet.activation.Identity;
import nnet.activation.Sigmoid;
import spawnn.SupervisedNet;
import spawnn.utils.DataFrame;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.transform;

public class NNet implements SupervisedNet {

	private static Logger log = Logger.getLogger(NNet.class);

	private Function[][] layer;
	private double[][][] weights;
	private double eta = 0.05;

	public NNet(Function[][] layer, double[][][] weights) {
		this.layer = layer;
		this.weights = weights;
	}
	
	public NNet(int[] l, boolean linout, double eta ) {
		Random r = new Random();
		
		// init layer
		this.layer = new Function[l.length][];
		for( int i = 0; i < l.length-1; i++ ) { // input or hidden layer
			this.layer[i] = new Function[l[i]+1]; // + bias
			for( int j = 0; j < l[i]; j++ )
				if( i == 0 ) // input 
					this.layer[i][j] = new Identity();
				else // hidden
					this.layer[i][j] = new Sigmoid();
			this.layer[i][l[i]] = new Constant(1.0); // bias					
		}
		this.layer[l.length-1] = new Function[l[l.length-1]]; // last/output layer
		for( int j = 0; j < l[l.length-1]; j++ )
			if( linout )  
				this.layer[l.length-1][j] = new Identity();
			else 
				this.layer[l.length-1][j] = new Sigmoid();	
		
		// init weights
		this.weights = new double[l.length-1][][];
		for( int i = 0; i < this.weights.length; i++ ) { 
			this.weights[i] = new double[this.layer[i].length][];
			for( int j = 0; j < this.weights[i].length; j++ ) {
				this.weights[i][j] = new double[l[i+1]]; // not to bias neurons
				for( int k = 0; k < this.weights[i][j].length; k++ ) 
					this.weights[i][j][k] = r.nextDouble();	
			}
		}
								
		this.eta = eta;
	}

	@Override
	public double[] present(double[] x) {
		double[] in = x;
		for (int i = 0; ; i++) {
			double[] out = new double[layer[i].length];
			// apply activation
			for (int j = 0; j < layer[i].length; j++)
				if (j < in.length) // extra-neurons are bias
					out[j] = layer[i][j].f(in[j]);
				else {
					out[j] = layer[i][j].f(0.0);
				}

			if (i == layer.length - 1)
				return out;

			// apply weights, calculate new input
			in = new double[weights[i][0].length]; // assumes first neuron is connected to all non-bias neurons
			for (int j = 0; j < out.length; j++)
				for (int k = 0; k < in.length; k++)
					in[k] += weights[i][j][k] * out[j];
		}
	}

	// t is not used
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
					in[k] += weights[i][j][k] *	out[i][j];
		}

		// back propagation
		double[][] error = new double[layer.length][];
		int ll = layer.length - 1; // index of last layer
		
		// get error
		error[ll] = new double[layer[ll].length];
		for (int j = 0; j < layer[ll].length; j++)
			error[ll][j] = (desired[j] - out[ll][j]) * layer[ll][j].fDevFOut(out[ll][j]);
		
		// change weights
		for (int j = 0; j < layer[ll - 1].length; j++)
			for (int k = 0; k < layer[ll].length; k++)
				weights[ll - 1][j][k] += eta * error[ll][k] * out[ll - 1][j]; // delta-rule
		
		for (int i = ll - 1; i > 0; i--) {
			// get error
			error[i] = new double[layer[i].length];
			for (int j = 0; j < layer[i].length; j++) {
				double s = 0;
				for (int k = 0; k < weights[i][j].length; k++)
					s += error[i + 1][k] * weights[i][j][k];
				error[i][j] = s * layer[i][j].fDevFOut(out[i][j]);
			}

			// change weights
			for (int j = 0; j < weights[i - 1].length; j++)
				for (int k = 0; k < weights[i - 1][j].length; k++)
					weights[i - 1][j][k] += eta * error[i][k] * out[i - 1][j];
		}
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
		Random r = new Random();

		DataFrame df = DataUtils.readDataFrameFromCSV(new File("data/airfoil_self_noise.csv"), new int[] {}, true);
		for (int i = 0; i < df.names.size(); i++)
			log.debug(i + ":" + df.names.get(i));
		DataUtils.transform(df.samples, transform.zScore);
		
		List<double[]> samples = new ArrayList<>();
		List<double[]> desired = new ArrayList<>();
		for (double[] d : df.samples) {
			samples.add(Arrays.copyOf(d, 5));
			desired.add(new double[] { d[5] });
		}
		
		DescriptiveStatistics dsRMSE = new DescriptiveStatistics(), dsR2 = new DescriptiveStatistics();
		List<Entry<List<Integer>,List<Integer>>> cvList = SupervisedUtils.getCVList(5, 5, df.samples.size());
		for( final Entry<List<Integer>,List<Integer>> cvEntry : cvList ) {
			List<double[]> samplesTrain = new ArrayList<double[]>();
			List<double[]> desiredTrain = new ArrayList<double[]>();
			for( int k : cvEntry.getKey() ) {
				samplesTrain.add(samples.get(k));
				desiredTrain.add(desired.get(k));
			}
			
			List<double[]> samplesVal = new ArrayList<double[]>();
			List<double[]> desiredVal = new ArrayList<double[]>();
			for( int k : cvEntry.getValue() ) {
				samplesVal.add(samples.get(k));
				desiredVal.add(desired.get(k));
			}
			
			NNet nnet = new NNet( new int[]{ samplesVal.get(0).length, 24, 1}, true, 0.01 );
			for (int i = 0; i < 100000; i++) {
				int idx = r.nextInt(samplesTrain.size());
				nnet.train(i, samplesTrain.get(idx), desiredTrain.get(idx));
			}
			
			List<double[]> response = new ArrayList<double[]>();
			for (double[] x : samplesVal)
				response.add(nnet.present(x));
			dsRMSE.addValue( SupervisedUtils.getRMSE(response, desiredVal) );
			dsR2.addValue( SupervisedUtils.getR2(response, desiredVal) );	
		}
		log.debug("RMSE: "+dsRMSE.getMean());
		log.debug("R2: "+dsR2.getMean());
	}
}
