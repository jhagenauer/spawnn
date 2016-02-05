package spawnn.ng.sorter;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;

public class SorterMNG extends SorterContext {
	
	protected Dist<double[]> dist;
	protected double alpha, beta;
	protected double[] last;

	public SorterMNG(Dist<double[]> dist, double alpha, double beta) {
		this.dist = dist;
		this.alpha = alpha;
		this.beta = beta;	
	}

	public void sort(final double[] x, List<double[]> neurons, double[] context) {
		Collections.sort(neurons, getComparator(x, context ) );
		last = neurons.get(0);
	}
	
	@Override
	public void sort(final double[] x, List<double[]> neurons) {
		Collections.sort(neurons, getComparator(x, getContext(x)) );
		last = neurons.get(0);
	}
	
	@Override
	public double[] getBMU( final double[] x, List<double[]> neurons ) {
		double[] bmu = Collections.min(neurons, getComparator(x, getContext(x)) );
		last = bmu;
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
	
	/*@Override
	public void sort(final double[] x, List<double[]> neurons) {	
		
		if( last != null ) { 
			context = new double[x.length];
			for( int i = 0; i < x.length; i++ )
				context[i] += (1 - beta) * last[i];
				
			for( int i = 0; i < x.length; i++ )
				context[i] += beta * last[i+x.length];
		} 
			
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
				
		last = neurons.get(0);
	}*/
	
	public double[] getContext(double[] x) { 
		double[] context = new double[x.length];
		for( int i = 0; i < x.length; i++ )
			context[i] += (1 - beta) * last[i];
				
		for( int i = 0; i < x.length; i++ )
			context[i] += beta * last[i+x.length]; 
		return context;
	}
	
	public void setLastBmu(double[] d) { 
		this.last = d;
	}
	
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}
	
	public void setBeta(double beta) {
		this.beta = beta;
	}
}
