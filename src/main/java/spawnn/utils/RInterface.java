package spawnn.utils;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import com.vividsolutions.jts.geom.Geometry;

import spawnn.dist.Dist;
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

	// Ugly, only for internal use/testing
	public static int[][] getCCC(double[][] samples, double[][] cMat, int[] fa1, int[] fa2, int[] numCluster, int threads) {
		

		int[][] fas = new int[][] { fa1, fa2 };
		String[] t = new String[] { "W", "A" };
		boolean[] scale = new boolean[] { false, true };

		ExecutorService es = Executors.newFixedThreadPool(threads);
		Future futures[][][] = new Future[scale.length][t.length][fas.length];

		for (int i = 0; i < scale.length; i++)
			for (int j = 0; j < t.length; j++)
				for (int k = 0; k < fas.length; k++) {

					final boolean sc = scale[i];
					final String tt = t[j];
					final int[] ffa = fas[k];

					futures[i][j][k] = es.submit(new Callable<int[][]>() {
						@Override
						public int[][] call() throws Exception {
							
							List<double[]> sa = new ArrayList<>();
							for( double[] d : samples )
								sa.add( Arrays.copyOf(d, d.length));
							
							if( sc )
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
							if (tt.equals("W"))
								type = HierarchicalClusteringType.ward;
							else if (tt.equals("A"))
								type = HierarchicalClusteringType.average_linkage;
							else if (tt.equals("C"))
								type = HierarchicalClusteringType.complete_linkage;
							else if (tt.equals("S"))
								type = HierarchicalClusteringType.single_linkage;

							List<TreeNode> tree = Clustering.getHierarchicalClusterTree(cm, new EuclideanDist(ffa), type);

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
					});
				}
		es.shutdown();

		try {
			int[][] ni = new int[scale.length * t.length * fas.length * numCluster.length][];
			for (int i = 0; i < scale.length; i++) 
				for (int j = 0; j < t.length; j++) 
					for( int k = 0; k < fas.length; k++ ){
						int[][] r = (int[][]) futures[i][j][k].get();
						for (int l = 0; l < numCluster.length; l++)
							ni[i * t.length * fas.length * numCluster.length + j * fas.length * numCluster.length + k * numCluster.length + l ] = r[l];					
					}
			return ni;
		} catch (InterruptedException ex) {
			ex.printStackTrace();
		} catch (ExecutionException ex) {
			ex.printStackTrace();
		}
		return null;
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

	public static void main(String[] args) {
		List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(new File("data/redcap/Election/election2004.shp"));
		List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/redcap/Election/election2004.shp"), new int[] {}, true);

		int[] fa = new int[] { 7 };
		DataUtils.transform(samples, fa, Transform.zScore);

		final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");
		final Dist<double[]> dist = new EuclideanDist(fa);

		int nrCluster = 7;

		double[][] s = new double[samples.size()][];
		for (int i = 0; i < samples.size(); i++)
			s[i] = samples.get(i);

		double[][] cMat = new double[s.length][s.length];
		for (int i = 0; i < s.length; i++)
			for (int j = 0; j < s.length; j++) {
				if (i == j)
					continue;
				double[] a = samples.get(i);
				double[] b = samples.get(j);
				if (cm.get(a).contains(b))
					cMat[i][j] = 1;
			}

		getCCC(s, cMat, new int[] { 1, 2 }, new int[] { 0, 2 }, new int[] { 3 }, 3);

	}
}
