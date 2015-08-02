package spawnn.ng.sorter;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;

public class SorterWMC extends SorterContext {
	
	private double[] context;
	
	protected Dist<double[]> dist;
	protected double alpha, beta;
	protected Map<double[],double[]> bmuHist; 
	protected Map<double[],Map<double[],Double>> weightMatrix;
		
	public boolean bmuHistMutable = false; 
	

	public SorterWMC(Map<double[],double[]> bmuHist, Map<double[],Map<double[],Double>> weightMatrix, Dist<double[]> dist, double alpha, double beta ) {
		this.dist = dist;
		this.alpha = alpha;
		this.beta = beta;	
		this.weightMatrix = weightMatrix;
		this.bmuHist = bmuHist;
	}
	
	@Override
	public void sort(final double[] x, List<double[]> neurons) {
		// get context
		context = getCurrentContext(x);
		
		// sort
		Collections.sort(neurons, new Comparator<double[]>() {
			@Override
			public int compare(double[] o1, double[] o2) {	
				double d1, d2;
				if( context != null ) {							
					d1 = (1-alpha) * dist.dist( o1, x ) + alpha * ((EuclideanDist)dist).dist( o1, x.length, context, 0 );
					d2 = (1-alpha) * dist.dist( o2, x ) + alpha * ((EuclideanDist)dist).dist( o2, x.length, context, 0 );
				} else {
					d1 = dist.dist( o1, x );
					d2 = dist.dist( o2, x );
				}
				return Double.compare(d1, d2);
			}
		});
		
		// update hist
		if( bmuHistMutable ) 
			bmuHist.put(x, neurons.get(0) );	
	}
		
	// Typically called by ContextNG
	@Override
	public double[] getContext(double[] x ) {
		return context; // a little faster and very slightly better results(?)
		//return getCurrentContext(x);
	}
	
	private double[] getCurrentContext(double[] x) {
		
		Map<double[],Double> nbs = weightMatrix.get(x);
		double[] context = new double[x.length];
		double sumWeights = 0;
		
		for( Entry<double[],Double> nb : nbs.entrySet() ) {
			double wij = nb.getValue(); 
			double[] bmuNb = bmuHist.get(nb.getKey());
			
			for( int k = 0; k < x.length; k++ )
				context[k] += wij * ( (1-beta) * bmuNb[k] + beta * bmuNb[x.length+k] );
			
			sumWeights += wij;
		}
				
		// normalize
		for( int k = 0; k < x.length; k++ )
			context[k] /= sumWeights;
					
		return context;	
	}
	
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}
	
	public double getAlpha() {
		return alpha;
	}
	
	public void setBeta(double beta) {
		this.beta = beta;
	}
	
	public void setHistMutable(boolean f) {
		bmuHistMutable = f;
	}
	
	public boolean histMutable() {
		return bmuHistMutable;
	}
}
