package aag_detroit;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import heuristics.CostCalculator;
import spawnn.dist.Dist;

public class ClusterCostCalculator implements CostCalculator<ClusterIndividual>{
	
	Map<Set<double[]>,Map<Set<double[]>,Double>> dm;
	List<Set<double[]>> prev;
	
	public ClusterCostCalculator(List<Set<double[]>> prev, List<Set<double[]>> ci, Dist<double[]> fDist ) {
		
		this.prev = prev;
		this.dm = new HashMap<>();
		for( Set<double[]> a : prev ) {
			Map<Set<double[]>,Double> ms = new HashMap<>();
			for( Set<double[]> b : ci ) {
				double di = 0;
				for( double[] d1 : a )
					for( double[] d2 : b )
						di += fDist.dist(d1, d2);
				di/=(a.size()*b.size());
				ms.put(b, di);
			}
			dm.put(a, ms);
		}
	}

	@Override
	public double getCost(ClusterIndividual i) {
		double di = 0;
		for( int j = 0; j < Math.min(prev.size(), i.getList().size()); j++ )
			di += dm.get(prev.get(j)).get(i.getList().get(j));
		return di;
	}
}
