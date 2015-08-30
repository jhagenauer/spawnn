package llm;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import spawnn.SupervisedNet;
import spawnn.ng.NG;
import spawnn.ng.sorter.Sorter;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;

// TODO It is better to separate NG and LLMNG, either completely or just give the ng as an argument to llm
// Question: Is LLMNG a neuralnet itself? What are the mappings?
// Separation:
// Pro:		
//		- Clean separation of sup und unsup training
// Contra:
//		- Is not independent of NG, because of sorter, neurons and range/rate
public class LLMNG extends NG implements SupervisedNet {
		
	public Map<double[],double[]> output = new HashMap<double[],double[]>(); // intercept
	public Map<double[],double[][]> matrix = new HashMap<double[],double[][]>(); // slope, jacobian matrix (m(output-dim) x n(input-dim) ), m rows, n columns
	
	private int[] fa;
	private DecayFunction neighborhoodRange2, adaptationRate2;
	private boolean ignoreSupport = false;
	
	public LLMNG( List<double[]> neurons, DecayFunction neighborhoodRange, DecayFunction adaptationRate,
			DecayFunction neighborhoodRange2, DecayFunction adaptationRate2, 
			Sorter<double[]> sorter, int[] fa, int outDim, boolean ignoreSupport ) {
		this(neurons,neighborhoodRange,adaptationRate,neighborhoodRange2,adaptationRate2,sorter,fa,outDim);
		this.ignoreSupport = ignoreSupport;
	}
	
	public LLMNG( List<double[]> neurons, DecayFunction neighborhoodRange, DecayFunction adaptationRate,
			DecayFunction neighborhoodRange2, DecayFunction adaptationRate2, 
			Sorter<double[]> sorter, int[] fa, int outDim ) {
		super(neurons,neighborhoodRange,adaptationRate,sorter);
		
		this.adaptationRate2 = adaptationRate2;
		this.neighborhoodRange2 = neighborhoodRange2;
				
		if( fa == null ) {
			this.fa = new int[neurons.get(0).length];
			for( int i = 0; i < this.fa.length; i++ )
				this.fa[i] = i;
		} else
			this.fa = fa;

		Random r = new Random();
		for( double[] d : getNeurons() ) {
			// init output
			double[] o = new double[outDim];
			for( int i = 0; i < o.length; i++ )
				o[i] = r.nextDouble();
			output.put( d, o );
			
			// init matrices
			double[][] m = new double[outDim][this.fa.length]; // m x n
			for (int i = 0; i < m.length; i++)
				for (int j = 0; j < m[i].length; j++)
					m[i][j] = r.nextDouble();
			matrix.put(d, m);
		}
	}
			
	@Deprecated
	public LLMNG( List<double[]> neurons, double lInit, double lFinal, double eInit, double eFinal,
			double lInit2, double lFinal2, double eInit2, double eFinal2, 
			Sorter<double[]> bg, int[] fa, int outDim ) {
		this(neurons, new PowerDecay(lInit, lFinal), new PowerDecay(eInit, eFinal), new PowerDecay(lInit2, lFinal2), new PowerDecay(eInit2, eFinal2), bg, fa, outDim);
	}
	
	public double[] present( double[] x ) {
		sorter.sort(x, neurons);
		return getResponse(x, getNeurons().get(0) );
	}
	
	public double[] getResponse( double[] x, double[] neuron ) {			
		double[][] m = matrix.get(neuron);
		double[] r = new double[m.length]; // calculate product m*diff
		
		for( int i = 0; i < m.length; i++ ) // row, outputs
			for( int j = 0; j < m[i].length; j++ ) // column, inputs
				if( !ignoreSupport )
					r[i] += m[i][j] * (x[fa[j]] - neuron[fa[j]] );
				else
					r[i] += m[i][j] * x[fa[j]];
				
		// add output
		for( int i = 0; i < r.length; i++ )
			r[i] += output.get(neuron)[i];
							
		return r;
	}
	
	public void train( double t, double[] x, double[] desired ) {
		train(t, x); // assumes that neurons are sorted after this call
		
		double l = neighborhoodRange2.getValue(t);
		double e = adaptationRate2.getValue(t);
				
		for( int k = 0; k < getNeurons().size(); k++ ) {
			double adapt = e * Math.exp( -(double)k/l );
			double[] w = getNeurons().get(k);
			
			// double[] r = getResponse( x, w ); 
			
			// adapt output
			double[] o = output.get(w); // output Vector
			for( int i = 0; i < desired.length; i++ ) { 
				o[i] += adapt * (desired[i] - o[i]); // martinetz
				//o[i] += adapt * (desired[i] - r[i]); // fritzke
			}
			
			double[] r = getResponse( x, w );
			
			// adapt matrix
			double[][] m = matrix.get(w);
			for( int i = 0; i < m.length; i++ )  // row
				for( int j = 0; j < m[i].length; j++ ) // column
					if( !ignoreSupport )
						m[i][j] += adapt * (desired[i] - r[i]) * (x[fa[j]] - w[fa[j]]); // outer product
					else
						m[i][j] += adapt * (desired[i] - r[i]) * x[fa[j]];
		}
	}
	
	public void setSorter( Sorter<double[]> sorter ) {
		this.sorter = sorter;
	}
}
