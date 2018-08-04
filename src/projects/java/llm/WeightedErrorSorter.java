package llm;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.log4j.Logger;

import spawnn.SupervisedNet;
import spawnn.dist.Dist;
import spawnn.ng.sorter.Sorter;

public class WeightedErrorSorter implements Sorter<double[]> {
	
	private static Logger log = Logger.getLogger(WeightedErrorSorter.class);
		
	SupervisedNet sn;
	int ta;
	Dist<double[]> dist;
	double weight;
	
	// Observations are mapped to different clusters if non-stationarity/heterogeneity.. test: are they?

	// Why not simply mapping solely by error?
	// What is the advantage?
	//		vs. k-means/ward + lm: cluster not only geo/var-defined but also relationship defined (depending on w, with w==1 --> llm ~ k-means/lm)
	//		vs. GWR: no clusters
	//		vs. GWR+ward: only coefficient-clusters
	// no definition of distance, or distance matrix.. cluster not necessarily spatial.. but they are if spatial dependent and spatial non-stationary
	// how does it work for a time series?
	
	// 
	
	// Features: Map coefficients, functional clusters
	// weight == 1 -> standard sorter
	// weight == 0 -> error sorter
	public WeightedErrorSorter( SupervisedNet sn, Dist<double[]> dist, List<double[]> samples, int ta, double weight) {
		this.sn = sn;
		this.dist = dist;
		this.weight = weight;
		this.ta = ta;
	}
	
	@Override
	public void sort( final double[] x, List<double[]> neurons ) {
		Map<double[],double[]> responses = new HashMap<>();
		for( double[] n : neurons )
			responses.put(n, sn.getResponse(x, n));
		
		Map<double[],Double> ranks = new HashMap<>();
		for( double[] n : neurons )
			ranks.put(n, 0.0);
		
		// sort by prototype 
		Comparator<double[]> c1 = new Comparator<double[]>() {
			@Override
			public int compare(double[] n1, double[] n2) {
				return Double.compare(dist.dist(n1, x), dist.dist(n2, x));
			}
		};
		Collections.sort(neurons,c1);
		for( int i = 0; i < neurons.size(); i++ ) {
			double[] n = neurons.get(i);
			ranks.put(n, weight*i );
		}
							
		// sort by error
		Comparator<double[]> c2 = new Comparator<double[]>() {
			@Override
			public int compare(double[] n1, double[] n2) {
							
				double[] d = new double[]{ x[ta] };
				double[] r1 = responses.get(n1);
				double[] r2 = responses.get(n2);
				
				double e1 = 0;
				for( int i = 0; i < r1.length; i++ )
					e1 += Math.pow(r1[i]-d[i],2);
				
				double e2 = 0;
				for( int i = 0; i < r2.length; i++ )
					e2 += Math.pow(r2[i]-d[i],2);
				
				return Double.compare(e1, e2);
			}
		};
		
		Collections.sort(neurons,c2);
		for( int i = 0; i < neurons.size(); i++ ) {
			double[] n = neurons.get(i);
			ranks.put(n, ranks.get(n) + (1.0-weight)*i );
		}
				
		// sort by final ranks
		Comparator<double[]> c3 = new Comparator<double[]>() {
			@Override
			public int compare(double[] n1, double[] n2) {
				return Double.compare( ranks.get(n1), ranks.get(n2) );
			}
		};
		Collections.sort(neurons,c3); 
	}
	
	public void setSupervisedNet( SupervisedNet sn ) {
		this.sn = sn;
	}

	@Override
	public double[] getBMU(double[] x, List<double[]> neurons) {
		sort(x,neurons);
		return neurons.get(0);
	}
}
