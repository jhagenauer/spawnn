package chowClustering;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;

import chowClustering.ChowClustering.ValSet;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;

public class Test_val {

	private static Logger log = Logger.getLogger(Test_val.class);

	public static int CLUST = 0, STRUCT_TEST = 1, P_VALUE = 2, DIST = 3, MIN_OBS = 4, PRECLUST = 5, PRECLUST_OPT = 6, PRECLUST_OPT2 = 7, MAJ_VOTE = 8;

	public static void main(String[] args) {

		int threads = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
		log.debug("Threads: " + threads);

		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/gemeinden_gs2010/gem_dat.shp"), new int[] { 1, 2 }, true);

		int[] ga = new int[] { 3, 4 };
		int[] fa = new int[] { 7, 8, 9, 10, 19, 20 };
		int ta = 18; // bldUpRt

		for (int i = 0; i < fa.length; i++)
			log.debug("fa " + i + ": " + sdf.names.get(fa[i]));
		log.debug("ta: " + ta + "," + sdf.names.get(ta));

		Dist<double[]> gDist = new EuclideanDist(ga);
		Map<double[], Set<double[]>> cm = GeoUtils.getContiguityMap(sdf.samples, sdf.geoms, false, false);

		Path file = Paths.get("output/chow.csv");
		try {
			Files.deleteIfExists(file);
			Files.createFile(file);
		} catch (IOException e1) {
			e1.printStackTrace();
		}

		int maxRun = 32;
		for (double p : new double[] { 0.1 }) {

			double[] results = new double[5];

			for (int r = 0; r < maxRun; r++) {
				ValSet vs = ChowClustering.getValSet(cm, p);

				List<Set<double[]>> init = ChowClustering.getInitCluster(vs.samplesTrain, vs.cmTrain, ChowClustering.PreCluster.Kmeans, 800, gDist);
				List<TreeNode> curLayer = new ArrayList<>();
				for (Set<double[]> s : init)
					curLayer.add(new TreeNode(0, 0, s));
				Map<TreeNode, Set<TreeNode>> ncm = ChowClustering.getCMforCurLayer(curLayer, vs.cmTrain);

				// HC 1, maintain minobs
				List<TreeNode> tree1 = Clustering.getHierarchicalClusterTree(curLayer, ncm, gDist, HierarchicalClusteringType.ward, fa.length + 2, threads);
				curLayer = Clustering.cutTree(tree1, 1);

				// HC 2
				// update curLayer/ncm
				for (TreeNode tn : curLayer)
					tn.contents = Clustering.getContents(tn);
				ncm = ChowClustering.getCMforCurLayer(curLayer, vs.cmTrain);

				List<TreeNode> tree = ChowClustering.getFunctionalClusterinTree(curLayer, ncm, fa, ta, HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.Wald, 1.0, threads);

				List<Set<double[]>> ct = Clustering.treeToCluster(Clustering.cutTree(tree, 12));
				LinearModel lm = new LinearModel(vs.samplesTrain, ct, fa, ta, false);

				// 0.05 0
				// 0.10 0
				// 0.15
				// 0.20
				for (int assignMode : new int[] { 0, 1, 2, 3, 4 }) {

					double rmse = 0;
					for (double[] d : vs.samplesVal) {
						Set<double[]> best = null;
						if (assignMode == 0) { // major Vote
							int bestCount = Integer.MAX_VALUE;
							for (Set<double[]> s : ct) {
								int count = 0;
								for (double[] nb : cm.get(d))
									if (s.contains(nb))
										count++;
								if (count > 0 && (best == null || count > bestCount)) {
									bestCount = count;
									best = s;
								}
							}
						} else if (assignMode == 1) {
							double[] closest = null; // get closest nb
							for (double[] nb : cm.get(d))
								if (closest == null || gDist.dist(d, nb) < gDist.dist(d, closest))
									closest = nb;

							// get s of closest nb
							for (Set<double[]> s : ct)
								if (s.contains(closest)) {
									best = s;
									break;
								}
						} else if (assignMode == 2) { // lowest inc of sse
							double bestInc = Double.POSITIVE_INFINITY;
							for (Set<double[]> s : ct) {

								boolean found = false;
								for (double[] nb : cm.get(d))
									if (s.contains(nb))
										found = true;
								if (!found)
									continue;

								double wssPre = DataUtils.getSumOfSquares(s, gDist);
								s.add(d);
								double inc = DataUtils.getSumOfSquares(s, gDist) - wssPre;
								s.remove(d);

								if (best == null || inc < bestInc) {
									best = s;
									bestInc = inc;
								}
							}
						} else if (assignMode == 3) { 

							double bestMean = Double.POSITIVE_INFINITY;
							for (Set<double[]> s : ct) {
								SummaryStatistics ss = new SummaryStatistics();
								for (double[] nb : cm.get(d))
									if (s.contains(nb))
										ss.addValue(gDist.dist(nb, d));
								if (ss.getN() > 0 && (best == null || ss.getMean() < bestMean)) {
									bestMean = ss.getMean();
									best = s;
								}
							}
						} else {
							double bestMean = Double.POSITIVE_INFINITY;
							for (Set<double[]> s : ct) {
								SummaryStatistics ss = new SummaryStatistics();
								for (double[] nb : cm.get(d))
									if (s.contains(nb))
										ss.addValue(Math.pow(gDist.dist(nb, d), 2));
								if (ss.getN() > 0 && (best == null || ss.getMean() < bestMean)) {
									bestMean = ss.getMean();
									best = s;
								}
							}
						}

						List<double[]> l = new ArrayList<>();
						l.add(d);

						best.add(d);
						double pred = lm.getPredictions(l, fa).get(0);
						best.remove(d);
						rmse += Math.pow(pred - d[ta], 2);
					}
					rmse = Math.sqrt(rmse / vs.samplesVal.size());
					results[assignMode] += rmse / maxRun;
				}
			}
			log.debug(p);
			for (int i = 0; i < results.length; i++)
				log.debug(i + "," + results[i]);
		}
	}

}
