package regionalization.ga;

import java.util.Set;

public interface ClusterCostCalculator {
	public double getCost(Set<double[]> cluster);
}
