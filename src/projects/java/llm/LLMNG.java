package llm;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import spawnn.SupervisedNet;
import spawnn.ng.NG;
import spawnn.ng.sorter.Sorter;
import spawnn.som.decay.DecayFunction;

public class LLMNG extends NG implements SupervisedNet {
		
	public Map<double[],double[]> output = new HashMap<double[],double[]>(); // intercept
	public Map<double[],double[][]> matrix = new HashMap<double[],double[][]>(); // slope, jacobian matrix (m(output-dim) x n(input-dim) ), m rows, n columns
	
	private int[] fa;
	private DecayFunction neighborhoodRange2, adaptationRate2;
	
	public enum mode {fritzke, martinetz};
		
	public LLMNG( List<double[]> neurons, 
			DecayFunction neighborhoodRange, DecayFunction adaptationRate,
			DecayFunction neighborhoodRange2, DecayFunction adaptationRate2, 
			Sorter<double[]> sorter, int[] fa, int outDim ) {
		super(neurons,neighborhoodRange,adaptationRate,sorter);
		
		this.adaptationRate2 = adaptationRate2;
		this.neighborhoodRange2 = neighborhoodRange2;
		this.fa = fa;

		Random r = new Random(1);
		for( double[] d : getNeurons() ) {
			// init output
			double[] o = new double[outDim];
			for( int i = 0; i < o.length; i++ )
				o[i] = r.nextDouble();
			output.put( d, o );
			
			// init matrices
			double[][] m = new double[outDim][this.neurons.get(0).length]; // m x n
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
	
	public double[] getResponse( double[] x, double[] w ) {			
		double[][] m = matrix.get(w);
		double[] r = new double[m.length]; // calculate product m*diff
		
		for( int i = 0; i < m.length; i++ ) // row, outputs
			for( int j = 0; j < fa.length; j++ ) // column, inputs
				if( ignSupport )
					r[i] += m[i][fa[j]] * x[fa[j]];
				else
					r[i] += m[i][fa[j]] * (x[fa[j]] - w[fa[j]] );
				
		// add output
		for( int i = 0; i < r.length; i++ )
			r[i] += output.get(w)[i];
							
		return r;
	}
	
	public mode aMode = mode.fritzke;
	public boolean ignSupport = false;
	
	@Override
	public void train( double t, double[] x, double[] desired ) {
		sorter.sort(x, neurons);
				
		double l = neighborhoodRange.getValue(t);
		double e = adaptationRate.getValue(t);
		double l2 = neighborhoodRange2.getValue(t);
		double e2 = adaptationRate2.getValue(t);

		for( int k = 0; k < getNeurons().size(); k++ ) {
			double[] w = getNeurons().get(k);
			double[] r = getResponse( x, w ); 
					
			// adapt prototype
			double adapt = e * Math.exp( -(double)k/l );
			for( int i = 0; i < w.length; i++ ) 
				w[i] +=  adapt * ( x[i] - w[i] );
									
			// adapt output
			double adapt2 = e2 * Math.exp( -(double)k/l2 );
			double[] o = output.get(w); // output Vector
			for( int i = 0; i < desired.length; i++ ) {
				if( aMode == mode.fritzke )
					o[i] += adapt2 * (desired[i] - o[i]);
				else if( aMode == mode.martinetz)
					o[i] += adapt2 * (desired[i] - r[i]);
			}
							
			// adapt matrix
			double[][] m = matrix.get(w);
			for( int i = 0; i < m.length; i++ )  // row
				for( int j = 0; j < x.length; j++ ) // column, nr of attributes
					if( ignSupport )
						m[i][j] += adapt2 * (desired[i] - r[i]) * x[j]; // outer product
					else
						m[i][j] += adapt2 * (desired[i] - r[i]) * (x[j] - w[j]); // outer product
		}
	}
	
	public void setSorter( Sorter<double[]> sorter ) {
		this.sorter = sorter;
	}
}
