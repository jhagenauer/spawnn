package spawnn.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;

public class RInterface {

	public static int[] getCNGCluster(double[][] samples, int numNeurons, double nbStart, double nbEnd, double lrStart, double lrEnd, int[] ga, int[] fa, int l, int trainingTime) {
		Random r = new Random();

		Sorter<double[]> s = new KangasSorter<double[]>(new EuclideanDist(ga), new EuclideanDist(fa), l);
		List<double[]> neurons = new ArrayList<double[]>();
		for (int i = 0; i < numNeurons; i++) {
			double[] rs = samples[r.nextInt(samples.length)];
			neurons.add(Arrays.copyOf(rs, rs.length));
		}

		NG ng = new NG(neurons, new PowerDecay(nbStart, nbEnd), new PowerDecay(lrStart, lrEnd), s);
		for (int t = 0; t < trainingTime; t++) {
			double[] x = samples[r.nextInt(samples.length)];
			ng.train((double) t / trainingTime, x);
		}
		
		List<double[]> n = new ArrayList<double[]>(neurons);
		int[] re = new int[samples.length];
		for( int i = 0; i < samples.length; i++ ) 
			re[i] = n.indexOf(s.getBMU(samples[i], neurons));
		return re;
	}

	public static int[][] getContConstCluster(double[][] samples, double[][] cMat, int[] fa, int[] numCluster, String t) {
		Map<double[], Set<double[]>> cm = new HashMap<double[], Set<double[]>>();
		for (int i = 0; i < samples.length; i++) {
			Set<double[]> s = new HashSet<>();
			for (int j = 0; j < cMat[i].length; j++)
				if (cMat[i][j] != 0)
					s.add(samples[j]);
			cm.put(samples[i], s);
		}
		
		HierarchicalClusteringType type = null;
		if (t.equals("W"))
			type = HierarchicalClusteringType.ward;
		else if (t.equals("A"))
			type = HierarchicalClusteringType.average_linkage;
		else if (t.equals("C"))
			type = HierarchicalClusteringType.complete_linkage;
		else if (t.equals("S"))
			type = HierarchicalClusteringType.single_linkage;

		List<TreeNode> tree = Clustering.getHierarchicalClusterTree(cm, new EuclideanDist(fa), type);
				
		int[][] r = new int[numCluster.length][samples.length];
		for( int i = 0; i < numCluster.length; i++ ) {
			List<Set<double[]>> cluster = Clustering.cutTree(tree, numCluster[i]);
			
			for( int j = 0; j < samples.length; j++ ) {
				for( int k = 0; k < cluster.size(); k++ ) {
					if( cluster.get(k).contains(samples[j]) )
						r[i][j] = k;
				}
			}
		}
				
		return r;
	}
}
