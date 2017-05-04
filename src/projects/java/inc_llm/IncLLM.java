package inc_llm;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import spawnn.SupervisedNet;
import spawnn.ng.sorter.Sorter;
import spawnn.som.decay.DecayFunction;

public class IncLLM implements SupervisedNet {
	
	protected List<double[]> neurons = null;
	protected double alpha, beta;
	protected Sorter<double[]> sorter;
	protected ConnectionsIncLLM cons;
	protected Map<double[],Double> errors;
	protected int aMax, lambda;
	protected int[] fa;
	
	public Map<double[],double[]> output = new HashMap<double[],double[]>(); // intercept
	public Map<double[],double[][]> matrix = new HashMap<double[],double[][]>(); // slope, jacobian matrix (m(output-dim) x n(input-dim) ), m rows, n columns
	
	private DecayFunction dfB, dfBln, dfN, dfNln;
	int t_max = 0;
	boolean useDf = false;
	
	// more than 2 neurons are removed immediately because they are not connected
	public IncLLM( Collection<double[]> neurons, DecayFunction dfB, DecayFunction dfBln, DecayFunction dfN, DecayFunction dfNln, Sorter<double[]> sorter, int aMax, int lambda, double alpha, double beta, int[] fa, int outDim, int t_max ) {
		this.dfB = dfB;
		this.dfBln = dfBln;
		this.dfN = dfN;
		this.dfNln = dfNln;
		this.useDf = true;
		this.t_max = t_max;
		
		this.cons = new ConnectionsIncLLM();
		this.aMax = aMax;
		this.lambda = lambda;
		this.alpha = alpha;
		this.beta = beta;
		this.sorter = sorter;
		this.neurons = new ArrayList<double[]>(neurons);
		this.fa = fa;
		
		this.errors = new HashMap<double[],Double>();
		for( double[] n : this.neurons )
			this.errors.put( n, 0.0 );
		
		Random r = new Random();
		for( double[] d : neurons ) {
			double[] o = new double[outDim];
			for( int i = 0; i < o.length; i++ )
				o[i] = r.nextDouble();
			output.put( d, o );
			
			double[][] m = new double[outDim][this.fa.length]; // m x n
			for (int i = 0; i < m.length; i++)
				for (int j = 0; j < m[i].length; j++)
					m[i][j] = r.nextDouble();
			matrix.put(d, m);
		}
	}
		
	public int maxNeurons = Integer.MAX_VALUE;
	
	private void train( double[] w, double[] x, double[] desired, double adapt, double adapt2 ) {
		double[] r = getResponse(x, w);
				
		// adapt prototype
		for( int i = 0; i < w.length; i++ )
			w[i] += adapt * ( x[i] - w[i] );
		
		// adapt output Vector
		double[] o = output.get(w); 
		for( int i = 0; i < desired.length; i++ )
			o[i] += adapt2 * (desired[i] - o[i]);
									
		// adapt matrix
		double[][] m = matrix.get(w);
		for( int i = 0; i < m.length; i++ )  // row
			for( int j = 0; j < fa.length; j++ ) // column, nr of attributes
				m[i][j] += adapt2 * (desired[i] - r[i]) * (x[fa[j]] - w[fa[j]]); // outer product
	}
		
	@Override
	public void train( double t, double[] x, double[] desired ) {
		double[] s_1 = sorter.getBMU(x, neurons);
		neurons.remove(s_1);
		double[] s_2 = sorter.getBMU(x, neurons);
		neurons.add(s_1);
				
		cons.add(s_1, s_2);
		
		double[] r = getResponse(x, s_1);
		double error = 0;
		for( int i = 0; i < desired.length; i++ )
			error += Math.pow( r[i] - desired[i], 2);
		errors.put(s_1, errors.get(s_1) + error );
		
		cons.increase(1);
		
		// train best neuron
		double nt = (double)t/t_max;
		train(s_1, x, desired, dfB.getValue(nt), dfBln.getValue(nt) );
		
		// train neighbors
		for( double[] n : cons.getNeighbors(s_1, 1) )
			train(n, x, desired, dfN.getValue(nt), dfNln.getValue(nt) );
				
		cons.purge(aMax);
				
		neurons.retainAll(cons.getVertices());
		errors.keySet().retainAll(cons.getVertices());
				
		if( (t+1) % lambda == 0 && neurons.size() < maxNeurons ) {
			
			double[] q = null;
			for( double[] n : neurons )
				if( q == null || errors.get(q) < errors.get(n) ) 
					q = n;
						
			double[] f = null;
			for( double[] n : cons.getNeighbors(q, 1) )
				if( f == null || errors.get(f) < errors.get(n) )
					f = n;
						
			double[] nn = new double[q.length];
			for( int i = 0; i < nn.length; i++ )
				nn[i] = (q[i]+f[i])/2;
			neurons.add(nn);
						
			cons.add( q, nn );
			cons.add( f, nn );
			cons.remove( q, f );
			
			errors.put(q, errors.get(q) - alpha*errors.get(q) );
			errors.put(f, errors.get(f) - alpha*errors.get(f) );
			
			errors.put(nn, (errors.get(q)+errors.get(f))/2);
			
			double[] nOutput = new double[output.get(q).length];
			for( int i = 0; i < nOutput.length; i++ )
				nOutput[i] = (output.get(q)[i] + output.get(f)[i])/2;
			output.put(nn, nOutput);
			
			double[][] nMatrix = new double[nOutput.length][fa.length];
			for( int i = 0; i < nOutput.length; i++ )
				for( int j = 0; j < fa.length; j++ )
					nMatrix[i][j] = (matrix.get(q)[i][j] + matrix.get(f)[i][j])/2;
			matrix.put(nn, nMatrix);
		}	
		
		for( double[] n : neurons )
			errors.put(n, errors.get(n) - beta*errors.get(n) );
	}
		
	public List<double[]> getNeurons() {
		return neurons;
	}
	
	@Override
	public double[] present( double[] x ) {
		return getResponse(x, sorter.getBMU(x, neurons) );
	}
	
	@Override
	public double[] getResponse( double[] x, double[] neuron ) {			
		double[][] m = matrix.get(neuron);
		double[] r = new double[m.length]; // calculate product m*diff
		
		for( int i = 0; i < m.length; i++ ) // row, outputs
			for( int j = 0; j < m[i].length; j++ ) // column, inputs
				r[i] += m[i][j] * (x[fa[j]] - neuron[fa[j]] );
				
		// add output
		for( int i = 0; i < r.length; i++ )
			r[i] += output.get(neuron)[i];
							
		return r;
	}
}
