package spawnn.ng.sorter;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;

public class SorterWMC extends SorterContext {

	protected Dist<double[]> dist;
	protected double alpha, beta;
	protected Map<double[],double[]> bmuHist; 
	protected Map<double[],Map<double[],Double>> weightMatrix;
		
	public boolean bmuHistMutable = true; // setting to immutable has not an effect anyways 
	
	public SorterWMC(Map<double[],double[]> bmuHist, Map<double[],Map<double[],Double>> weightMatrix, Dist<double[]> dist, double alpha, double beta ) {
		this.dist = dist;
		this.alpha = alpha;
		this.beta = beta;	
		this.weightMatrix = weightMatrix;
		this.bmuHist = bmuHist;
	}
		
	public double getDist( double[] x, double[] neuron ) {	
		double[] context = getContext(x);
		if( context != null )						
			return (1-alpha) * dist.dist( neuron, x ) + alpha * ((EuclideanDist)dist).dist( neuron, x.length, context, 0 );
		else 
			return dist.dist( neuron, x );
	}
	
	public void sort(final double[] x, List<double[]> neurons, double[] context) {
		Collections.sort(neurons, getComparator(x, context ) );
		// update hist
		double[] bmu = neurons.get(0);
		if( bmuHistMutable && bmuHist.get(x) != bmu ) {
			bmuHist.put(x, bmu );
		}
	}
	
	@Override
	public void sort(final double[] x, List<double[]> neurons) {
		Collections.sort(neurons, getComparator(x, getContext(x)) );
		// update hist
		double[] bmu = neurons.get(0);
		if( bmuHistMutable && bmuHist.get(x) != bmu ) {
			bmuHist.put(x, bmu );
		}
	}
	
	@Override
	public double[] getBMU( final double[] x, List<double[]> neurons ) {
		double[] bmu = Collections.min(neurons, getComparator(x, getContext(x)) );
		// update hist
		if( bmuHistMutable && bmuHist.get(x) != bmu ) {
			bmuHist.put(x, bmu );
		}
		return bmu;
	}
	
	private Comparator<double[]> getComparator( final double[] x, final double[] context ) {
		return new Comparator<double[]>() {
			@Override
			public int compare(double[] n1, double[] n2) {	
				double d1 = (1-alpha) * dist.dist( n1, x ) + alpha * ((EuclideanDist)dist).dist( n1, x.length, context, 0 );
				double d2 = (1-alpha) * dist.dist( n2, x ) + alpha * ((EuclideanDist)dist).dist( n2, x.length, context, 0 );
				return Double.compare(d1, d2);
			}
		};
	}
		
	@Override
	public double[] getContext(double[] x) {
		double[] context = new double[x.length];
		double sumWeights = 0;
				
		for( Entry<double[],Double> nb : weightMatrix.get(x).entrySet() ) {
			double wij = nb.getValue(); 
			double[] bmuNb = bmuHist.get(nb.getKey());
						
			for( int k = 0; k < x.length; k++ )
				context[k] += wij * ( (1-beta) * bmuNb[k] + beta * bmuNb[x.length+k] );
			
			sumWeights += wij;
		}
						
		// normalize
		if( sumWeights > 0.0 ) 
			for( int k = 0; k < x.length; k++ )
				context[k] /= sumWeights;
		
		return context;	
	}
		
	public void setHistMutable(boolean f) {
		bmuHistMutable = f;
	}
	
	public boolean histMutable() {
		return bmuHistMutable;
	}
	
	public void setWeightMatrix( Map<double[],Map<double[],Double>> weightMatrix ) {
		this.weightMatrix = weightMatrix;
	}
}
