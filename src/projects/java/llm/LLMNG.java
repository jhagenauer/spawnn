package llm;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import spawnn.SupervisedNet;
import spawnn.ng.NG;
import spawnn.ng.sorter.Sorter;

// TODO It is better to separate NG and LLMNG, either completly or just give the ng as an argument to llm
// Also make an interface or somethin for supervised learning
// Question: Is LLMNG a neuralnet itself? What are the mappings?
public class LLMNG extends NG implements SupervisedNet {
		
	public Map<double[],double[]> output = new HashMap<double[],double[]>(); // intercept
	public Map<double[],double[][]> matrix = new HashMap<double[],double[][]>(); // slope, jacobian matrix (m(output-dim) x n(input-dim) ), m rows, n columns
	
	private double lInit2, lFinal2, eInit2, eFinal2;
	private int[] fa;
	
	@Deprecated
	public LLMNG( int numNeurons, double lInit, double lFinal, double eInit, double eFinal, int dim, Sorter<double[]> bg, int outDim ) {
		this(numNeurons,lInit,lFinal,eInit,eFinal,lInit,lFinal,eInit,eFinal, bg, null, dim, 1);
	}
	
	@Deprecated
	public LLMNG( int numNeurons, double lInit, double lFinal, double eInit, double eFinal,
			double lInit2, double lFinal2, double eInit2, double eFinal2, 
			Sorter<double[]> bg, int[] fa, int inDim, int outDim ) {
		super(numNeurons,lInit,lFinal,eInit,eFinal,inDim,bg);
		
		this.lInit2 = lInit2;
		this.lFinal2 = lFinal2;
		this.eInit2 = eInit2;
		this.eFinal2 = eFinal2;
		
		if( fa == null ) {
			this.fa = new int[inDim];
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
	
	public LLMNG( List<double[]> neurons, double lInit, double lFinal, double eInit, double eFinal,
			double lInit2, double lFinal2, double eInit2, double eFinal2, 
			Sorter<double[]> bg, int[] fa, int outDim ) {
		super(neurons,lInit,lFinal,eInit,eFinal,bg);
		
		this.lInit2 = lInit2;
		this.lFinal2 = lFinal2;
		this.eInit2 = eInit2;
		this.eFinal2 = eFinal2;
		
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
	
	public double[] present( double[] x ) {
		sorter.sort(x, neurons);
		return getResponse(x, getNeurons().get(0) );
	}
	
	public double[] getResponse( double[] x, double[] neuron ) {			
		double[][] m = matrix.get(neuron);
		double[] r = new double[m.length]; // calculate product m*diff
		
		for( int i = 0; i < m.length; i++ ) // row, outputs
			for( int j = 0; j < m[i].length; j++ ) // column, inputs
				r[i] += m[i][j] * (x[fa[j]] - neuron[fa[j]]);
				
		// add output
		for( int i = 0; i < r.length; i++ )
			r[i] += output.get(neuron)[i];
							
		return r;
	}
	
	public void train( double t, double[] x, double[] desired ) {
		train(t, x); // assumes that neurons are sorted after this call
		
		double l = lInit2 * Math.pow( lFinal2/lInit2, t );
		double e = eInit2 * Math.pow( eFinal2/eInit2, t );
				
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
					m[i][j] += adapt * (desired[i] - r[i]) * (x[fa[j]] - w[fa[j]]); // outer product
		}
	}
	
	public void setSorter( Sorter<double[]> sorter ) {
		this.sorter = sorter;
	}
}
