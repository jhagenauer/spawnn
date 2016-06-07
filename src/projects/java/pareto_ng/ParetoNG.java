package pareto_ng;


import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import spawnn.UnsupervisedNet;
import spawnn.dist.Dist;
import spawnn.som.decay.DecayFunction;

public class ParetoNG implements UnsupervisedNet {
	
	protected List<double[]> neurons = null;
	protected Dist<double[]> distA, distB;
	protected DecayFunction neighborhoodRange, adaptationRate;
		
	public ParetoNG( List<double[]> neurons, DecayFunction neighborhoodRange, DecayFunction adaptationRate, Dist<double[]> distA, Dist<double[]> distB  ) {
		this.distA = distA;
		this.distB = distB;
		this.neurons = new ArrayList<double[]>(neurons);
		this.neighborhoodRange = neighborhoodRange;
		this.adaptationRate = adaptationRate;
	}
				
	public void train( double t, double[] x ) {
		List<Set<double[]>> fronts = getParetoFronts(x);
		
		double l = neighborhoodRange.getValue(t);
		double e = adaptationRate.getValue(t);
		
		for( int k = 0; k < fronts.size(); k++ ) {
			double adapt = e * Math.exp( -(double)k/l );
			for( double[] w : fronts.get(k) )
				for( int i = 0; i < w.length; i++ ) 
					w[i] +=  adapt * ( x[i] - w[i] ) ;
		} 
		return;
	}
		
	public List<double[]> getNeurons() {
		return neurons;
	}
	
	//TODO improve performance, lots of potential
	public List<Set<double[]>> getParetoFronts( double[] x ) {
		List<Set<double[]>> fronts = new ArrayList<Set<double[]>>();
		
		Set<double[]> open = new HashSet<double[]>(neurons);
		while( !open.isEmpty() ) {
			
			Set<double[]> front = new HashSet<double[]>();			
			for( double[] a : open ) {
				double distAa = distA.dist(a, x);
				double distBa = distB.dist(a, x);
				
				boolean pareto = true;
				for( double[] b : open ) {
					if( a == b )
						continue;
					
					double distAb = distA.dist(b,x);
					double distBb = distB.dist(b,x);
					// b dominates a
					if(  (distAb < distAa && distBb <= distBa ) ||  (distAb <= distAa && distBb < distBa ) ) {
						pareto = false;
						break;
					}
				}
				if( pareto )
					front.add(a);
			}
			fronts.add(front);
			open.removeAll(front);
		}	
		return fronts;
	}
}
