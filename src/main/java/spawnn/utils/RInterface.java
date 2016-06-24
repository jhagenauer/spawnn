package spawnn.utils;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import spawnn.dist.EuclideanDist;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;

public class RInterface {
	
	public static int[] getContConstCluster( double[][] samples, double[][] cMat, int[] fa, int numCluster, String t ) {
		Map<double[],Set<double[]>> cm = new HashMap<double[],Set<double[]>>();
		for( int i = 0; i < samples.length; i++ ) {
			Set<double[]> s = new HashSet<double[]>();
			for( int j = 0; j < cMat[i].length; j++ )
				if( j != 0 )
					s.add( samples[j] );
			cm.put(samples[i], s);
		}
		
		HierarchicalClusteringType type = null;
		if( t.equals("W") )
			type = HierarchicalClusteringType.ward;
		else if( t.equals("A") )
			type = HierarchicalClusteringType.average_linkage;
		else if( t.equals("C") )
			type = HierarchicalClusteringType.complete_linkage;
		else if( t.equals("S") )
			type = HierarchicalClusteringType.single_linkage;
				
		List<TreeNode> tree = Clustering.getHierarchicalClusterTree(cm, new EuclideanDist(fa), type );
		List<Set<double[]>> cluster = Clustering.cutTree(tree, numCluster);
		int[] r = new int[samples.length];
		for( int i = 0; i < samples.length; i++ ) 
			for( int j = 0; j < cluster.size(); j++ )
				if( cluster.get(j).contains(samples[i] ) ) {
					r[i] = j;
					break;
				}
		return r;
	}
}
