package inc_llm;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import spawnn.SupervisedNet;
import spawnn.ng.Connection;
import spawnn.ng.sorter.Sorter;

public class IncLLM implements SupervisedNet {
	
	protected List<double[]> neurons = null;
	protected double lrB, lrBln, lrN, lrNln, alpha, beta;
	protected Sorter<double[]> sorter;
	protected Map<Connection,Integer> cons;
	protected Map<double[],Double> errors;
	protected int aMax, lambda;
	protected int[] fa;
	
	public Map<double[],double[]> output = new HashMap<double[],double[]>(); // intercept
	public Map<double[],double[][]> matrix = new HashMap<double[],double[][]>(); // slope, jacobian matrix (m(output-dim) x n(input-dim) ), m rows, n columns
	
		
	public IncLLM( Collection<double[]> neurons, double lrB, double lrBln, double lrN, double lrNln, Sorter<double[]> sorter, int aMax, int lambda, double alpha, double beta, int[] fa, int outDim ) {
		this.lrB = lrB;
		this.lrBln = lrBln;
		this.lrN = lrN;
		this.lrNln = lrNln;
		this.cons = new HashMap<Connection,Integer>();
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
		for( double[] d : getNeurons() ) {
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
	
	private void train( double[] neuron, double[] x, double[] desired, double a, double aLn ) {
		double[] response = getResponse(x, neuron);
		// adapt prototype
		for( int i = 0; i < neuron.length; i++ )
			neuron[i] += a * ( x[i] - neuron[i] );
		
		// adapt output Vector
		double[] o = output.get(neuron); 
		for( int i = 0; i < desired.length; i++ )
			o[i] += aLn * (desired[i] - o[i]);
									
		// adapt matrix
		double[][] m = matrix.get(neuron);
		for( int i = 0; i < m.length; i++ )  // row
			for( int j = 0; j < m[i].length; j++ ) // column, nr of attributes
				m[i][j] += aLn * (desired[i] - response[i]) * (x[fa[j]] - neuron[fa[j]]); // outer product
	}
		
	@Override
	public void train( double t, double[] x, double[] desired ) {
		sorter.sort(x, neurons);
		double[] s_1 = neurons.get(0);
		double[] s_2 = neurons.get(1);
		
		cons.put(new Connection(s_1, s_2),0);
		
		double[] r = getResponse(x, s_1);
		double error = 0;
		for( int i = 0; i < desired.length; i++ )
			error += Math.pow( r[i] - desired[i], 2);
		errors.put(s_1, errors.get(s_1) + error );
		
		for( Connection c : cons.keySet() )
			cons.put(c, cons.get(c)+1 );
		
		// train best neuron
		train(s_1, x, desired, lrB, lrBln);
		
		// train neighbors
		for( double[] n : Connection.getNeighbors(cons.keySet(), s_1, 1) )
			train(n, x, desired, lrN, lrNln);
		
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
		
		if( t % lambda == 0 && neurons.size() < maxNeurons ) {
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
	
	public Map<Connection, Integer> getConections() {
		return cons;
	}

	@Override
	public double[] present( double[] x ) {
		sorter.sort(x, neurons);
		return getResponse(x, getNeurons().get(0) );
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
