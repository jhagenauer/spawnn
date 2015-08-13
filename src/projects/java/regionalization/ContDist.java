package regionalization;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import spawnn.dist.Dist;
import spawnn.utils.RegionUtils;

public class ContDist implements Dist<double[]> {
	
	Map<double[],Integer> idxMap;
	int[][] dist;
	
	public ContDist( final List<double[]> samples, final Map<double[], Set<double[]>> cm ) {
		this.dist = RegionUtils.getDistMatrix(cm, samples);
		
		this.idxMap = new HashMap<double[],Integer>();
		for( int i = 0; i < samples.size(); i++ )
			idxMap.put(samples.get(i), i);
	}

	@Override
	public double dist(double[] a, double[] b) {
		int idxA = idxMap.get(a);
		int idxB = idxMap.get(b);
		
		int d = 0;
		if( idxA < idxB )
			d = dist[idxA][idxB];
		else
			d = dist[idxB][idxA];
		return d;
	}
}
