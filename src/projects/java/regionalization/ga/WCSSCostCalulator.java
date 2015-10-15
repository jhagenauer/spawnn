package regionalization.ga;

import java.util.Set;

import spawnn.dist.Dist;
import spawnn.utils.DataUtils;

public class WCSSCostCalulator implements ClusterCostCalculator {
	
	Dist<double[]> fDist;
	
	public WCSSCostCalulator(Dist<double[]> fDist ) {
		this.fDist = fDist;
	}

	@Override
	public double getCost(Set<double[]> cluster) {
		return DataUtils.getSumOfSquares(cluster, fDist);
	}
}
