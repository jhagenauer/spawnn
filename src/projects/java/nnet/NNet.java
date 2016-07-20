package nnet;

import java.util.Random;

import org.apache.log4j.Logger;

import nnet.activation.Constant;
import nnet.activation.Function;
import nnet.activation.SoftMax;
import spawnn.SupervisedNet;

public class NNet implements SupervisedNet {

	private static Logger log = Logger.getLogger(NNet.class);

	private Function[][] layer;
	private double[][][] weights;
	private double eta = 0.05;
	
	public NNet(Function[][] layer, double[][][] weights) {
		this.layer = layer;
		this.weights = weights;
	}
	
	public NNet(Function[][] l, double eta ) {
		Random r = new Random();
		
		this.eta = eta;
		this.layer = l;
		
		// init weights
		this.weights = new double[this.layer.length][][];
		for( int i = 0; i < this.weights.length-1; i++ ) { 
			
			this.weights[i] = new double[this.layer[i].length][];
			for( int j = 0; j < this.weights[i].length; j++ ) {
				
				int notConstant = 0;
				for( int k = 0; k < this.layer[i+1].length; k++ )
					notConstant += (this.layer[i+1][k] instanceof Constant) ? 1 : 0; 
				this.weights[i][j] = new double[layer[i+1].length-notConstant];
				
				for( int k = 0; k < this.weights[i][j].length; k++ ) 
					this.weights[i][j][k] = r.nextDouble()-0.5;	
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
			int idxIn = 0;
			for (int j = 0; j < layer[i].length; j++ )
				if( layer[i][j] instanceof SoftMax )
					out[j] = ((SoftMax)layer[i][j]).f(in,idxIn++);
				else
					out[j] = layer[i][j] instanceof Constant ? layer[i][j].f(1.0) : layer[i][j].f(in[idxIn++]);
						
			if (i == layer.length - 1)
				return out;

			// apply weights, calculate new input
			in = new double[weights[i][0].length]; // assumes neuron 0 is connected to all non-bias neurons
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
			int idxIn = 0;
			for (int j = 0; j < layer[i].length; j++)
				if( layer[i][j] instanceof SoftMax )
					out[i][j] = ((SoftMax)layer[i][j]).f(in,idxIn++);
				else
					out[i][j] = layer[i][j] instanceof Constant ? layer[i][j].f(0) : layer[i][j].f(in[idxIn++]);
							
			if (i == layer.length - 1)
				break;
			
			// apply weights, calculate new net input
			in = new double[weights[i][0].length]; // assumes neuron 0 is connected to all non-bias neurons
			for (int j = 0; j < out[i].length; j++)
				for (int k = 0; k < in.length; k++)
					in[k] += weights[i][j][k] *	out[i][j];
		}

		// back propagation
		double[][] error = new double[layer.length][];
		int ll = layer.length - 1; // index of last layer
		
		// get error of last layer
		error[ll] = new double[layer[ll].length];
		for (int j = 0; j < layer[ll].length; j++)
			if( layer[ll][j] instanceof SoftMax ) {
				error[ll][j] = out[ll][j] - desired[j]; 
			} else
				error[ll][j] = (out[ll][j] - desired[j]) * layer[ll][j].fDevFOut(out[ll][j]);
		
		// change weights of last layer
		for (int j = 0; j < layer[ll - 1].length; j++)
			for (int k = 0; k < layer[ll].length; k++)
				weights[ll - 1][j][k] -= eta * error[ll][k] * out[ll - 1][j]; // delta-rule
		
		// get error and change weights of hidden layer
		for (int i = ll - 1; i > 0; i--) {
			// get error
			error[i] = new double[layer[i].length];
			for (int j = 0; j < layer[i].length; j++) {
				
				double s = 0;
				for (int k = 0; k < weights[i][j].length; k++)
					s += error[i + 1][k] * weights[i][j][k];
				
				if( layer[i][j] instanceof SoftMax )
					throw new RuntimeException("SoftMax not allowed in hidden layer!");
				else
					error[i][j] = s * layer[i][j].fDevFOut(out[i][j]);
			}

			// change weights
			for (int j = 0; j < weights[i - 1].length; j++)
				for (int k = 0; k < weights[i - 1][j].length; k++)
					weights[i - 1][j][k] -= eta * /*(double)(ll-i)/ll **/ error[i][k] * out[i - 1][j];
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
}
