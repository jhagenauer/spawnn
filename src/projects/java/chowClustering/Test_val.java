package chowClustering;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;

import chowClustering.ChowClustering.ValSet;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.ClusterValidation;
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

		int threads = Math.max(1, Runtime.getRuntime().availableProcessors()-1);
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

		int kmCluster = 300;
		int nrCluster = 12;
		int runs = 8;

		for (double p : new double[] { 0.1 }) {
			log.debug("p: " + p);

			Map<ValSet,List<Set<double[]>>> vsList = new HashMap<>();
			for (int i = 0; i < runs; i++) {
				ValSet vs = ChowClustering.getValSet(cm, p);
				
				List<Set<double[]>> init = ChowClustering.getInitCluster(sdf.samples, cm, ChowClustering.PreCluster.Kmeans, kmCluster, gDist);
				List<TreeNode> curLayer = new ArrayList<>();
				for (Set<double[]> s : init)
					curLayer.add(new TreeNode(0, 0, s));
				Map<TreeNode, Set<TreeNode>> ncm = ChowClustering.getCMforCurLayer(curLayer, cm);

				// HC 1, maintain minobs
				List<TreeNode> tree1 = Clustering.getHierarchicalClusterTree(curLayer, ncm, gDist, HierarchicalClusteringType.ward, fa.length + 2, threads);
				curLayer = Clustering.cutTree(tree1, 1);
				
				// HC 2
				// update curLayer/ncm
				for (TreeNode tn : curLayer)
					tn.contents = Clustering.getContents(tn);
				ncm = ChowClustering.getCMforCurLayer(curLayer, cm);

				List<TreeNode> tree = ChowClustering.getFunctionalClusterinTree(curLayer, ncm, fa, ta, HierarchicalClusteringType.ward, ChowClustering.StructChangeTestMode.Wald, 1.0, threads);
				List<Set<double[]>> ref = Clustering.treeToCluster(Clustering.cutTree(tree, nrCluster));
				log.debug(ref.size()+", "+ClusterValidation.getWithinClusterSumOfSuqares(ref, gDist));
				
				vsList.put(vs, ref);
			}

			for (boolean minMode : new boolean[] { true, false }) {
				Clustering.r.setSeed(0);
				log.debug(minMode);
				Clustering.minMode = minMode;

				double[] nmis = new double[9];
				double[] rmses = new double[9];

				for (Entry<ValSet,List<Set<double[]>>> e1 : vsList.entrySet()) {
					ValSet vs = e1.getKey();
					List<Set<double[]>> ref = e1.getValue();

					List<Set<double[]>> init = ChowClustering.getInitCluster(vs.samplesTrain, vs.cmTrain, ChowClustering.PreCluster.Kmeans, kmCluster, gDist);
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
					List<Set<double[]>> ct = Clustering.treeToCluster(Clustering.cutTree(tree, nrCluster));
					log.debug(ct.size()+", "+ClusterValidation.getWithinClusterSumOfSuqares(ct, gDist));
					
					for (int assignMode : new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 }) {

						Map<Set<double[]>, Set<double[]>> m = new HashMap<>();

						for (double[] d : vs.samplesVal) {
							Set<double[]> best = null;
							if( assignMode == 0 ) { // major vote
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
							} else if( assignMode == 1 ) { // closest gDist
								double[] closest = null;
								for (Set<double[]> s : ct) {
									for( double[] nb : cm.get(d) )
										if( s.contains(nb) && ( best == null || gDist.dist(nb, d) < gDist.dist(closest, d) ) ) {
											best = s;
											closest = nb;
										}
								}
							} else if( assignMode == 2 ) { // SSE-Diff
								double bestInc = Double.POSITIVE_INFINITY;
								for (Set<double[]> s : ct) {
									for (double[] nb : cm.get(d))
										if (s.contains(nb)) {
											double wssPre = DataUtils.getSumOfSquares(s, gDist);
											s.add(d);
											double inc = DataUtils.getSumOfSquares(s, gDist) - wssPre;
											s.remove(d);
											
											if (best == null || inc < bestInc) {
												best = s;
												bestInc = inc;
											}
										}
								}
							} else if( assignMode == 3 ) { // mean gDist
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
							} else if( assignMode == 4 ) { // mean square gDist
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
							} else if (assignMode == 5) { // major vote + mean square gDist
								int bestCount = -1;
								double bestMean = Double.POSITIVE_INFINITY;
								for (Set<double[]> s : ct) {
									SummaryStatistics ss = new SummaryStatistics();
									int count = 0;
									for (double[] nb : cm.get(d))
										if (s.contains(nb)) {
											count++;
											ss.addValue(Math.pow(gDist.dist(nb, d), 2));
										}
									if (count == 0)
										continue;

									if (best == null || count > bestCount || (count == bestCount && ss.getMean() < bestMean)) {
										bestCount = count;
										bestMean = ss.getMean();
										best = s;
									}
								}
							} else if (assignMode == 6) { // major vote + gDist
								int bestCount = -1;
								double bestMean = Double.POSITIVE_INFINITY;
								for (Set<double[]> s : ct) {
									SummaryStatistics ss = new SummaryStatistics();
									int count = 0;
									for (double[] nb : cm.get(d))
										if (s.contains(nb)) {
											count++;
											ss.addValue(gDist.dist(nb, d));
										}
									if (count == 0)
										continue;

									if (best == null || count > bestCount || (count == bestCount && ss.getMean() < bestMean)) {
										bestCount = count;
										bestMean = ss.getMean();
										best = s;
									}
								}
							} else if (assignMode == 7) { // major vote + min gDist
								int bestCount = -1;
								double bestMean = Double.POSITIVE_INFINITY;
								for (Set<double[]> s : ct) {
									double dist = Double.MAX_VALUE;
									int count = 0;
									for (double[] nb : cm.get(d))
										if (s.contains(nb)) {
											count++;
											dist = Math.min(gDist.dist(d, nb), dist);
										}
									if (count == 0)
										continue;

									if (best == null || count > bestCount || (count == bestCount && dist < bestMean)) {
										bestCount = count;
										bestMean = dist;
										best = s;
									}
								}
							} else if( assignMode == 8 ) { // major vote + min SSE
								int bestCount = -1;
								double bestInc = Double.POSITIVE_INFINITY;
								for (Set<double[]> s : ct) {
									double inc = Double.MAX_VALUE;
									int count = 0;
									for (double[] nb : cm.get(d))
										if (s.contains(nb)) {
											count++;
											
											double wssPre = DataUtils.getSumOfSquares(s, gDist);
											s.add(d);
											inc = DataUtils.getSumOfSquares(s, gDist) - wssPre;
											s.remove(d);
										}
									if (count == 0)
										continue;

									if (best == null || count > bestCount || (count == bestCount && inc < bestInc)) {
										bestCount = count;
										bestInc = inc;
										best = s;
									}
								}
							}

							if (!m.containsKey(best))
								m.put(best, new HashSet<>());
							m.get(best).add(d);
						}

						for (Entry<Set<double[]>, Set<double[]>> e : m.entrySet())
							e.getKey().addAll(e.getValue());

						double nmi = ClusterValidation.getNormalizedMutualInformation(ct, ref);
						nmis[assignMode] += nmi / runs;

						LinearModel lm = new LinearModel(vs.samplesTrain, ct, fa, ta, false);
						List<Double> pred = lm.getPredictions(vs.samplesVal, fa);
						double rmse = Math.sqrt(SupervisedUtils.getMSE(pred, vs.samplesVal, ta));
						rmses[assignMode] += rmse / runs;

						for (Entry<Set<double[]>, Set<double[]>> e : m.entrySet())
							e.getKey().removeAll(e.getValue());
					}
				}
				log.debug(p);
				for (int i = 0; i < nmis.length; i++)
					log.debug(i + "," + nmis[i] + "\t" + rmses[i]);
			}
		}
	}

}
