package spawnn.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import spawnn.dist.EuclideanDist;
import spawnn.ng.NG;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils.Transform;

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
		for (int i = 0; i < samples.length; i++)
			re[i] = n.indexOf(s.getBMU(samples[i], neurons));
		return re;
	}
		
	public static Map<int[],int[][]> getHC(double[][] samples, double[][] cMat, int[] fa, boolean[] constrained, boolean[] scale, String[] cType, int[] numCluster, int threads) {
		
		ExecutorService es = Executors.newFixedThreadPool(threads);
		Map<int[],Future<int[][]>> futures = new HashMap<>();

		for( int i = 0; i < constrained.length; i++ )
		for( int j = 0; j < scale.length; j++ )
		for( int k = 0; k < cType.length; k++ ) {
			
			final boolean c = constrained[i];
			final boolean s = scale[j]; 
			final String ct = cType[k];
		
					futures.put(new int[]{i,j,k}, es.submit(new Callable<int[][]>() {
						@Override
						public int[][] call() throws Exception {
							
							List<double[]> sa = new ArrayList<>();
							for( double[] d : samples )
								sa.add( Arrays.copyOf(d, d.length));

							if( s )
								DataUtils.transform(sa, Transform.zScore);
							
							Map<double[], Set<double[]>> cm = new HashMap<double[], Set<double[]>>();
							for (int i = 0; i < sa.size(); i++) {
								Set<double[]> s = new HashSet<>();
								for (int j = 0; j < cMat[i].length; j++)
									if (cMat[i][j] != 0)
										s.add(sa.get(j));
								cm.put(sa.get(i), s);
							}
														
							HierarchicalClusteringType type = null;
							if (ct.equals("W"))
								type = HierarchicalClusteringType.ward;
							else if (ct.equals("A"))
								type = HierarchicalClusteringType.average_linkage;
							else if (ct.equals("C"))
								type = HierarchicalClusteringType.complete_linkage;
							else if (ct.equals("S"))
								type = HierarchicalClusteringType.single_linkage;

							List<TreeNode> tree;
							if( c )
								tree = Clustering.getHierarchicalClusterTree(cm, new EuclideanDist(fa), type);
							else
								tree = Clustering.getHierarchicalClusterTree(sa, new EuclideanDist(fa), type);

							int[][] r = new int[numCluster.length][sa.size()];
							for (int i = 0; i < numCluster.length; i++) {
								List<Set<double[]>> cluster = Clustering.cutTree(tree, numCluster[i]);
								for (int j = 0; j < sa.size(); j++) {
									for (int k = 0; k < cluster.size(); k++) {
										if (cluster.get(k).contains(sa.get(j)))
											r[i][j] = k;
									}
								}
							}
							return r;
						}
					}));
				}
		es.shutdown();
		
		Map<int[],int[][]> r = new HashMap<>();
		try {
			for( Entry<int[],Future<int[][]>> e : futures.entrySet() ) 
				r.put(e.getKey(), e.getValue().get() );
		} catch (InterruptedException ex) {
			ex.printStackTrace();
		} catch (ExecutionException ex) {
			ex.printStackTrace();
		}
		return r;
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
		for (int i = 0; i < numCluster.length; i++) {
			List<Set<double[]>> cluster = Clustering.cutTree(tree, numCluster[i]);

			for (int j = 0; j < samples.length; j++) {
				for (int k = 0; k < cluster.size(); k++) {
					if (cluster.get(k).contains(samples[j]))
						r[i][j] = k;
				}
			}
		}

		return r;
	}
}
