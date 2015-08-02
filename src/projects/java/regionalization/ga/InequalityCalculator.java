package regionalization.ga;

import java.util.Set;

public class InequalityCalculator implements ClusterCostCalculator {
	
	int[] fa;
	double mean;
	
	public InequalityCalculator( int[] fa, double mean ) {
		this.fa = fa;
		this.mean = mean;
	}

	@Override
	public double getCost(Set<double[]> cluster) {
		double sum = 0;
		for( double[] d : cluster )
			sum += d[fa[0]];				
		return Math.abs(sum - mean);
	}
}
