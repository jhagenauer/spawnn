package wmng.llm;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import spawnn.SupervisedNet;
import spawnn.ng.NG;
import spawnn.ng.sorter.SorterContext;
import spawnn.ng.sorter.SorterWMC;
import spawnn.som.decay.DecayFunction;

public class ContextNG_LLM extends NG implements SupervisedNet {
	
	public Map<double[],double[]> output = new HashMap<double[],double[]>(); // intercept
	public Map<double[],double[][]> matrix = new HashMap<double[],double[][]>(); // slope, jacobian matrix (m(output-dim) x n(input-dim) ), m rows, n columns
	
	private int[] fa;
	private DecayFunction neighborhood2, adaptation2;
	
	public ContextNG_LLM( List<double[]> neurons, 
			DecayFunction neighborhood, DecayFunction adaptation, 
			DecayFunction neighborhood2, DecayFunction adaptation2, 
			SorterContext sorter, int[] fa, int outDim ) {
		super(neurons,neighborhood,adaptation,sorter);
		
		this.adaptation2 = adaptation2;
		this.neighborhood2 = neighborhood2;
		this.fa = fa;
		
		Random r = new Random();
		for( double[] d : getNeurons() ) {
			// init output
			double[] o = new double[outDim];
			for( int i = 0; i < o.length; i++ )
				o[i] = r.nextDouble();
			output.put( d, o );
			
			// init matrices
			double[][] m = new double[outDim][neurons.get(0).length]; // m x n
			for (int i = 0; i < m.length; i++)
				for (int j = 0; j < m[i].length; j++)
					m[i][j] = r.nextDouble();
			matrix.put(d, m);
		}
	}
	
	@Override
	public double[] present(double[] x) {
		return getResponse(x, sorter.getBMU(x, neurons) );
	}
	
	public boolean useCtx = true;

	@Override
	public double[] getResponse(double[] x, double[] w) {
		return getResponse(x, ((SorterContext)sorter).getContext(x), w);
	}
	
	private double[] getResponse(double[] x, double[] context, double[] w) {
		double[][] m = matrix.get(w);
		double[] r = new double[m.length]; // calculate product m*diff
						
		for( int i = 0; i < m.length; i++ ) // row, outputs
			for( int j = 0; j < fa.length; j++ ) // column, inputs
				r[i] += m[i][fa[j]] * ( x[fa[j]] - w[fa[j]] );
					
		if( useCtx )
		for( int i = 0; i < m.length; i++ ) // row, outputs
			for( int j = 0; j < fa.length; j++ ) // column, inputs
				r[i] += m[i][x.length + fa[j]] * ( context[fa[j]] - w[x.length + fa[j]] );
									
		// add output
		for( int i = 0; i < r.length; i++ )
			r[i] += output.get(w)[i];
							
		return r;
	}

	@Override
	public void train(double t, double[] x, double[] desired) {
		// sort affects context
		double[] context = ((SorterContext)sorter).getContext(x);
		if( sorter instanceof SorterWMC )
			((SorterWMC)sorter).sort(x, neurons, context);
		else
			sorter.sort(x, neurons);
		
		double l = neighborhoodRange.getValue(t);
		double e = adaptationRate.getValue(t);
		double l2 = neighborhood2.getValue(t);
		double e2 = adaptation2.getValue(t);
		
		// adapt
		for (int k = 0; k < neurons.size(); k++) {
			double[] w = neurons.get(k);
			double[] r = getResponse( x, context, w ); 
			
			double adapt = e * Math.exp((double) -k / l);			
			// adapt weights and context vector part
			for (int i = 0; i < x.length; i++) {
				w[i] += adapt * (x[i] - w[i]);
				w[x.length + i] += adapt * (context[i] - w[x.length + i]);
			}
						
			// -------------------------------------------------------------------------
			
			// adapt output
			double adapt2 = e2 * Math.exp( -(double)k/l2 );
			double[] o = output.get(w); // output Vector
			for( int i = 0; i < desired.length; i++ )
				o[i] += adapt2 * (desired[i] - o[i]);
							
			// adapt matrix
			double[][] m = matrix.get(w);
			for( int i = 0; i < m.length; i++ )  // row
				for( int j = 0; j < x.length; j++ ) { // column, attributes
					m[i][j] += adapt2 * (desired[i] - r[i]) * (x[j] - w[j]);
					m[i][x.length + j] += adapt2 * (desired[i] - r[i]) * (context[j] - w[x.length + j]);
				}	
		}
	}
	
	// just for debug
	public void contextNGLLMtoString() {
		System.out.println("Prototypes: ");
		for (double[] n : neurons)
			System.out.println("w: " + n[2] + ", wc: " + n[6]);
		System.out.println("m:");
		for (double[] n : neurons) {
			double[] m = matrix.get(n)[0];
			System.out.println("m: " + m[2] + ", mc:" + m[6]);
		}
	}

}
