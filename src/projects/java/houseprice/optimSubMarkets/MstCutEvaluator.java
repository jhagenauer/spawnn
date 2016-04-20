package houseprice.optimSubMarkets;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import myga.Evaluator;
import spawnn.dist.Dist;

public class MstCutEvaluator implements Evaluator<MaintainSumIndividual> {
	private List<double[]> samplesTrain, desiredTrain, samplesVal, desiredVal;
	private int[] fa;
	private Dist<double[]> dist;
	
	private Map<double[],Set<double[]>> mst;
	private List<Entry<double[],double[]>> cuts;

	MstCutEvaluator( List<double[]> samplesTrain, List<double[]> desiredTrain, List<double[]> samplesVal, List<double[]> desiredVal, int[] fa, Dist<double[]> dist, Map<double[],Set<double[]>> mst, List<Entry<double[],double[]>> cuts ) {
		this.samplesTrain = samplesTrain;
		this.desiredTrain = desiredTrain;
		this.samplesVal = samplesVal;
		this.desiredVal = desiredVal;
		this.fa = fa;
		this.dist = dist;
		
		this.mst = mst;
		this.cuts = cuts;
	}
	
	public static Set<Set<double[]>> getContiguousClusters( List<double[]> samples, Map<double[],Set<double[]>> cm ) {
		Set<double[]> ds = new HashSet<double[]>(samples);		
		Set<Set<double[]>> all = new HashSet<Set<double[]>>();
		while( !ds.isEmpty() ) {
			double[] d = ds.iterator().next(); 
			Set<double[]> sub = getContiguousCluster( d, cm ); 	
			all.add(sub);
			ds.removeAll(sub);
		}
		return all;
	}
	
	public static Set<double[]> getContiguousCluster( double[] d, Map<double[],Set<double[]>> cm ) {
		Set<double[]> visited = new HashSet<double[]>();
		List<double[]> openList = new ArrayList<double[]>();
		openList.add( d );
			
		while( !openList.isEmpty() ) {
			double[] cur = openList.remove(openList.size()-1);
			visited.add(cur);
				
			// get all neighbors
			Set<double[]> nbs = new HashSet<double[]>();
			if( cm.containsKey(cur))
				nbs.addAll(cm.get(cur));
			for( Entry<double[],Set<double[]>> e : cm.entrySet() )
				if( e.getValue().contains(cur) )
					nbs.add(e.getKey());
			
			// get neighbors not visited and add to openList
			for( double[] nb : nbs )
				if( !visited.contains( nb ) )
					openList.add( nb );
		}
		return visited;
	}
	
	@Override
	public double evaluate(MaintainSumIndividual msi) {
		Map<double[],Set<double[]>> cutTree = new HashMap<double[],Set<double[]>>();
		for( Entry<double[],Set<double[]>> e : mst.entrySet() )
			cutTree.put(e.getKey(), new HashSet<double[]>(e.getValue()) );
		
		for( int i = 0; i < cuts.size(); i++ )
			if( msi.getChromosome()[i] ) {
				Entry<double[],double[]> e = cuts.get(i);
				cutTree.get(e.getKey()).remove(e.getValue());
			}
		
		List<Set<double[]>> cluster = new ArrayList<Set<double[]>>(getContiguousClusters(samplesTrain,cutTree));
						
		// add vals to cluster
		Map<double[],Set<double[]>> bestCs = new HashMap<double[],Set<double[]>>();
		for( double[] d : samplesVal ) {
			Set<double[]> bestC = null;
			double bestDist = Double.MAX_VALUE;
			for( Set<double[]> c : cluster )
				for( double[] dd : c )
					if( bestC == null || dist.dist(dd, d) < bestDist ) {
						bestDist = dist.dist(dd, d);
						bestC = c;
					}
			bestCs.put(d,bestC);
		}
		for( Entry<double[],Set<double[]>> e : bestCs.entrySet() )
			e.getValue().add(e.getKey());
								
		return SubmarketsByClusterLM.getRMSEofLM(cluster, samplesTrain, desiredTrain, samplesVal, desiredVal, fa);
	}
}
