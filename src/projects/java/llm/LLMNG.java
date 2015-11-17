package llm;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.log4j.Logger;

import spawnn.SupervisedNet;
import spawnn.ng.NG;
import spawnn.ng.sorter.Sorter;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;

public class LLMNG extends NG implements SupervisedNet {
	
	private static Logger log = Logger.getLogger(LLMNG.class);
		
	public Map<double[],double[]> output = new HashMap<double[],double[]>(); // intercept
	public Map<double[],double[][]> matrix = new HashMap<double[],double[][]>(); // slope, jacobian matrix (m(output-dim) x n(input-dim) ), m rows, n columns
	
	private int[] fa;
	private DecayFunction neighborhoodRange2, adaptationRate2;
	
	public enum mode {fritzke, martinetz,hagenauer};
		
	public LLMNG( List<double[]> neurons, 
			DecayFunction neighborhoodRange, DecayFunction adaptationRate,
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
			
			// int last-lists
			lastX.put(d, new LinkedList<double[]>());
			lastDesired.put(d,new LinkedList<double[]>());
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
				if( ignSupport )
					r[i] += m[i][j] * x[fa[j]];
				else
					r[i] += m[i][j] * (x[fa[j]] - neuron[fa[j]] );
				
		// add output
		for( int i = 0; i < r.length; i++ )
			r[i] += output.get(neuron)[i];
							
		return r;
	}
	
	public mode aMode = mode.fritzke;
	public boolean ignSupport = false;
		
	Map<double[],LinkedList<double[]>> lastX = new HashMap<double[],LinkedList<double[]>>();
	Map<double[],LinkedList<double[]>> lastDesired = new HashMap<double[],LinkedList<double[]>>();

	@Override
	public void train( double t, double[] x, double[] desired ) {
		sortNeurons(x);
				
		double l = neighborhoodRange.getValue(t);
		double e = adaptationRate.getValue(t);
		double l2 = neighborhoodRange2.getValue(t);
		double e2 = adaptationRate2.getValue(t);

		double[] lnSys = new double[fa.length+1];
		lnSys[lnSys.length-1] = desired[0]; 
		double det = 1;
		
		for( int k = 0; k < getNeurons().size(); k++ ) {
			double[] w = getNeurons().get(k);
			double[] r = getResponse( x, w ); 
					
			// adapt prototype
			double adapt = e * Math.exp( -(double)k/l );
			for( int i = 0; i < w.length; i++ ) 
				w[i] +=  adapt * ( x[i] - w[i] );
						
			if( aMode == mode.hagenauer ) {	
				if( k == 0 && lastX.get(w).size() == fa.length ) {
					try {
						
						RealMatrix coefficients = new Array2DRowRealMatrix(fa.length+1,fa.length+1);
						RealVector constants = new ArrayRealVector(fa.length+1);
											
						for( int j = 0; j < fa.length; j++ ) // x
							coefficients.setEntry(0, j, x[fa[j]]);
						coefficients.setEntry(0, fa.length, 1); // intercept
						constants.setEntry(0,desired[0]);
											
						for( int i = 0; i < lastX.get(w).size(); i++ ) { // last xs
							for( int j = 0; j < fa.length; j++ )
								coefficients.setEntry(i+1, j, lastX.get(w).get(i)[fa[j]]);
							coefficients.setEntry(i+1, fa.length, 1); // intercept
							constants.setEntry(i+1, lastDesired.get(w).get(i)[0]);
						}
						LUDecomposition decomp = new LUDecomposition(coefficients);
						lnSys = decomp.getSolver().solve(constants).toArray();	
						
						RealMatrix m1 = new Array2DRowRealMatrix(fa.length,fa.length);
						for( int i = 0; i < fa.length; i++ )
							for( int j = 0; j < fa.length; j++ )
								m1.setEntry(i, j, lastX.get(w).get(i)[fa[j]]-x[fa[j]]);							
						det = new LUDecomposition(m1).getDeterminant();					
					} catch( SingularMatrixException ex ) { 
						//log.debug(ex.getMessage());
					}
				} 
						
				// fifo
				lastX.get(w).addFirst(x);
				lastDesired.get(w).addFirst(desired);
				
				while( lastX.get(w).size() > fa.length ) {
					lastDesired.get(w).removeLast();
					lastX.get(w).removeLast();
				}
			}
						
			double d = lnSys[lnSys.length-1]; // calc intercept at x
			if( aMode == mode.hagenauer )
				for( int j = 0; j < fa.length; j++ )
					d += x[fa[j]] * lnSys[j];
						
			// adapt output and matrix
			double adapt2 = e2 * Math.exp( -(double)k/l2 );
			double[] o = output.get(w); // output Vector
			for( int i = 0; i < desired.length; i++ ) {
				if( aMode == mode.fritzke )
					o[i] += adapt2 * (desired[i] - o[i]);
				else if( aMode == mode.martinetz)
					o[i] += adapt2 * (desired[i] - r[i]);
				else { // hagenauer
					o[i] += adapt2 * (d - o[i]);
				}
			}
							
			// adapt matrix
			double[][] m = matrix.get(w);
			for( int i = 0; i < m.length; i++ )  // row
				for( int j = 0; j < m[i].length; j++ ) // column, nr of attributes
					if( aMode == mode.hagenauer ) {
						m[i][j] += adapt2 * (lnSys[j] - m[i][j]) * Math.abs(det);							
					} else {
						if( ignSupport )
							m[i][j] += adapt2 * (desired[i] - r[i]) * x[fa[j]]; // outer product
						else
							m[i][j] += adapt2 * (desired[i] - r[i]) * (x[fa[j]] - w[fa[j]]); // outer product
					}
		}
	}
	
	public void setSorter( Sorter<double[]> sorter ) {
		this.sorter = sorter;
	}
}
