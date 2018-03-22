package aag_detroit;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import heuristics.CostCalculator;
import spawnn.dist.Dist;
import spawnn.utils.DataUtils;

public class ClusterCostCalculator implements CostCalculator<SAClusterIndividual> {
	
	Map<Set<double[]>,Map<Set<double[]>,Double>> dm;
	List<Set<double[]>> prev;
	
	public ClusterCostCalculator(List<Set<double[]>> prev, List<Set<double[]>> ci, Dist<double[]> fDist ) {
		if( prev.size() > ci.size() ) {
			System.err.println("prev.size() should be <= ci.size()");
			System.exit(1);
		}
		
		this.prev = prev;
		this.dm = new HashMap<>();
		for( Set<double[]> a : prev ) {
			double[] mA = DataUtils.getMean(a);
			Map<Set<double[]>,Double> ms = new HashMap<>();			
			for( Set<double[]> b : ci ) {
				double[] mB = DataUtils.getMean(b);
				double di = fDist.dist(mA, mB);					
				ms.put(b, di);
			}
			dm.put(a, ms);
		}
	}

	@Override
	public double getCost( SAClusterIndividual ci) {
		double di = 0;
		for( int j = 0; j < prev.size(); j++ )
			di += dm.get( prev.get(j) ).get( ci.getList().get(j) );
		return di;
	}
	
	public void printCosts( SAClusterIndividual ci ) {
		for( int i = 0; i < prev.size(); i++ )
			for( int j = 0; j < ci.getList().size(); j++ )
				System.out.println((i+1)+", "+(j+1)+" -> "+dm.get(prev.get(i)).get(ci.getList().get(j))+"\t"+prev.get(i).size()+"::"+ci.getList().get(j).size());
	}
}
